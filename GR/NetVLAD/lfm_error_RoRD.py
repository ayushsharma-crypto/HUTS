import argparse
import numpy as np
from NetVLAD.dataset import get_whole_val_set
import re
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.linalg import logm
from pyquaternion import Quaternion
from typing import Tuple, List, Dict
from tabulate import tabulate
import seaborn as sns

parser = argparse.ArgumentParser(description='PLOTTING THE RETRIEVAL')
parser.add_argument('--root_dir', type=str, default='', help='Path to dataset')
parser.add_argument('--dataset', type=str, default='berlin', 
        help='Dataset to use', choices=['small_2','spair_90', 'spair_180','SMALL','TOPVIEW','PERSPECTIVE','sT4fr6TAbpF','oxford', 'nordland', 'berlin'])
parser.add_argument('--netvlad_predictions', type=str, default='', help='Path to NetVLAD Predictions')
parser.add_argument('--reference_poses', type=str, default='', help='Path to reference_poses that needs to be filtered')
parser.add_argument('--query_poses', type=str, default='', help='Path to query_poses that needs to be filtered')
parser.add_argument('--rord_trans', type=str, default='', help='Path to predicted transformation computed using RoRD')
parser.add_argument('--rord_match_count', type=str, default='', help='Path to rord match count computed using RoRD')
parser.add_argument('--name', type=str, default='', help='Name to plot')


def quaternion_inverse(given_quat):
    """
    Input: scipy format - scalar LAST. numpy.array([b, c, d, a]) where quaternion is a + bi + cj + dk
    or in wxyz format, w is scalar below
    Output: numpy array in same format but inverse of input
    """
    #pyquaternion library: Scalar FIRST: numpy.array([a, b, c, d]) where quaternion is a + bi + cj + dk
    qx, qy, qz, qw = given_quat #scalar last
    given_quat_scalar_first = np.array([qw, qx, qy, qz]) #scalar first
    quat_obj = Quaternion(array=given_quat_scalar_first) 
    inv_quat = quat_obj.inverse
    inv_quat_np = inv_quat.elements

    iqw, iqx, iqy, iqz  = inv_quat_np #scalar first
    inv_quat_np_scalar_first = np.array([iqx, iqy, iqz, iqw]) #scalar  last
    return  inv_quat_np_scalar_first


def get_rotation_error_using_arccos_of_trace_of_rotation(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    # See https://www.notion.so/saishubodh/Evaluation-Metrics-for-R-t-with-GT-Rotation-Translation-error-with-Ground-truth-476db686933048d5b74de36b382e0eec
        # measure the angular distance between two rotation matrice
    # R1,R2: [n, 3, 3]
    # Returns angle in degrees
    if R_pred.shape == (3,3):
        R_pred = R_pred[np.newaxis,:]
    if R_gt.shape == (3,3):
        R_gt = R_gt[np.newaxis,:]
    n = R_gt.shape[0]
    trace_idx = [0,4,8]
    trace = np.matmul(R_pred, R_gt.transpose(0,2,1)).reshape(n,-1)[:,trace_idx].sum(1)
    metric_in_degrees = np.arccos(((trace - 1)/2).clip(-1,1)) 
    metric_in_degrees = metric_in_degrees / np.pi * 180.0
    return metric_in_degrees[0]


def get_rotation_error_using_quaternion_dot(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    # See https://www.notion.so/saishubodh/Evaluation-Metrics-for-R-t-with-GT-Rotation-Translation-error-with-Ground-truth-476db686933048d5b74de36b382e0eec
    # cpp
    # const float d1 = std::fabs(q1.dot(q2));
    #d2 = std::fmin(1.0f, std::fmax(-1.0f, d1))
    #return 2 * acos(d2) * 180 / M_PI

    # return np.linalg.norm(logm(np.matmul(R_pred, R_gt.transpose()))) / np.sqrt(2)
    if R_pred.shape != (3, 3) or R_gt.shape != (3, 3):
        raise ValueError(f'Input matrices must be 3x3, instead got {R_pred.shape} and {R_gt.shape}')

    R_pred_obj, R_gt_obj = R.from_matrix(R_pred), R.from_matrix(R_gt)
    quat_pred, quat_gt = R_pred_obj.as_quat(), R_gt_obj.as_quat()

    # quat_pred_inv, quat_gt_inv = quaternion_inverse(quat_pred), quaternion_inverse(quat_gt)
    # quat_pred_inv_i, quat_gt_inv_i = quaternion_inverse(quat_pred_inv), quaternion_inverse(quat_gt_inv)
    # print(quat_pred, quat_gt)
    # print(quat_pred_inv, quat_gt_inv)
    # print(quat_pred_inv_i, quat_gt_inv_i)

    # d1 = abs(np.dot(quat_pred, quat_gt))
    # print("Quat inverse")
    d1 = abs(np.dot(quaternion_inverse(quat_pred), quat_gt))
    d2 = np.min((1.0, np.max((-1.0, d1))))
    rotation_error_in_degrees = 2 * np.arccos(d2) * 180 / np.pi
    # print(rotation_error_in_degrees)
    return rotation_error_in_degrees

def get_rotation_error_using_log_of_rotation(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    # See https://www.notion.so/saishubodh/Evaluation-Metrics-for-R-t-with-GT-Rotation-Translation-error-with-Ground-truth-476db686933048d5b74de36b382e0eec
    if R_pred.shape != (3, 3) or R_gt.shape != (3, 3):
        raise ValueError(f'Input matrices must be 3x3, instead got {R_pred.shape} and {R_gt.shape}')

    return np.linalg.norm(logm(np.matmul(R_pred, R_gt.transpose()))) / np.sqrt(2)

def get_translation_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    if t_gt.shape != (3, ) or t_gt.shape != (3, ):
        raise ValueError(f'Input vectors must be 3 dimensional, instead got {t_pred.shape} and {t_gt.shape}')

    return np.linalg.norm(t_gt - t_pred)

def get_both_errors(T_pred: np.ndarray, T_gt: np.ndarray) -> Tuple[float, float]:
    if T_pred.shape != (4, 4) or T_gt.shape != (4, 4):
        raise ValueError(f'Input matrices must be 4x4, instead got {T_pred.shape} and {T_gt.shape}')

    # rot_error = get_rotation_error_using_quaternion_dot(T_pred[:3, :3], T_gt[:3, :3])
    # print(rot_error, "quat")
    rot_error = get_rotation_error_using_arccos_of_trace_of_rotation(T_pred[:3, :3], T_gt[:3, :3])
    # print(rot_error, "trace")
    trans_error = get_translation_error(T_pred[:3, -1], T_gt[:3, -1])
    return (rot_error, trans_error)



def plot_line(rot_errors, trans_errors):
    LABL = [  "trans. err", "rot. err"]
    plt.plot(range(len(rot_errors)),trans_errors)
    plt.plot(range(len(rot_errors)),rot_errors)
    plt.xlabel("Query IDX")
    plt.ylabel("errors")
    plt.legend(LABL, loc="upper right")
    plt.yticks(np.arange(0, 30, step=2))
    plt.xticks(np.arange(0, 100, step=10))
    plt.title("Rot. & Trans. error Vs Query Index")
    plt.show()


def get_errors(preds: List[np.ndarray], truths: List[np.ndarray], query, retrieved) -> Tuple[List[float], List[float]]:
    if len(preds) != len(truths):
        raise ValueError(f'Input must consist of equal numbers of predictions and ground truths, instead got {len(preds)} and {len(truths)}')
    
    rot_errors = []
    trans_errors = []

    idx=0

    for T_pred, T_gt in zip(preds, truths):
        rot_err, trans_err = get_both_errors(T_pred, T_gt)
        rot_errors.append(rot_err)
        trans_errors.append(trans_err)
        idx += 1
        print(idx, query[idx-1], retrieved[idx-1], rot_errors[-1], trans_errors[-1])
            # print("\n\nT_pred = \n", T_pred, "\n\nT_gt = \n",T_gt)
    # plot_line(rot_errors, trans_errors)
    return rot_errors, trans_errors


def filter_err(rot_err, trans_err, rot_thresh, trans_thresh):
    r_idx = np.argwhere(rot_err<=rot_thresh)
    t_idx = np.argwhere(trans_err<=trans_thresh)
    idx = np.intersect1d(r_idx, t_idx)
    return len(idx)/len(rot_err)


def print_statistics(data: List[Dict]) -> None:
    table = []
    for dataset in data:
        res = {
            'Dataset': dataset['name']+":mean",
            'Rot. Error': np.mean(dataset['rot_errors']),
            'Trans. Error': np.mean(dataset['trans_errors']),
        }
        table.append(res)
        res = {
            'Dataset': dataset['name']+":max",
            'Rot. Error': np.max(dataset['rot_errors']),
            'Trans. Error': np.max(dataset['trans_errors']),
        }
        table.append(res)
        res = {
            'Dataset': dataset['name']+":min",
            'Rot. Error': np.min(dataset['rot_errors']),
            'Trans. Error': np.min(dataset['trans_errors']),
        }
        table.append(res)

    # print results
    print(tabulate(table, headers='keys', tablefmt='presto'))

    # histogram
    # fig, axs = plt.subplots(len(data), 2)
    # axs = axs.reshape((len(data), 2)) # when len(data) = 1
    # for i in range(len(data)):
    #     r = data[i]['rot_errors']
    #     t = data[i]['trans_errors']
        
    #     axs[i, 0].hist(r, weights=np.ones(len(r))/len(r))
    #     axs[i, 0].set_title(data[i]['name'])
    #     axs[i, 0].set(xlabel='Rot. Error', ylabel='Fraction')

    #     axs[i, 1].hist(t, weights=np.ones(len(t))/len(t))
    #     axs[i, 1].set_title(data[i]['name'])
    #     axs[i, 1].set(xlabel='Trans. Error', ylabel='Fraction')

    # plt.tight_layout()
    # plt.show()


    # making 2-D combined plot
    bins_translation = np.arange(3, 19, 3)
    bins_rotation = np.arange(30, 181, 30)
    RP,TP, FREQ = [], [], []
    for r in bins_rotation:
        for t in bins_translation:
            FREQ.append(filter_err(dataset['rot_errors'], dataset['trans_errors'], r, t))
            RP.append(r)
            TP.append(t)
    

    CDF = np.array(FREQ).reshape((len(bins_rotation), len(bins_translation)))
    # print(len(RP), len(TP), len(FREQ), len(CDF))

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    # print(CDF)
    PDF = np.empty(CDF.shape)
    for r in range(len(bins_rotation)):
        for c in range(len(bins_translation)):
            PDF[r,c]=CDF[r,c]
            if(r>0):
                PDF[r,c] -= CDF[r-1,c]
            if(c>0):
                PDF[r,c] -= CDF[r,c-1]
            if(r>0 and c>0):
                PDF[r,c] += CDF[r-1,c-1]


    df = pd.DataFrame({
        'rotation-precision': RP,
        'translation-precision': TP,
        'Recall/query-percentage': FREQ
    })
    df = df.pivot('rotation-precision', 'translation-precision', 'Recall/query-percentage')
    ax = sns.heatmap(df, linewidths=.5, annot=True)
    plt.show()


    df = pd.DataFrame({
        'rotation-precision': RP,
        'translation-precision': TP,
        'Recall/query-percentage': PDF.flatten()
    })
    df = df.pivot('rotation-precision', 'translation-precision', 'Recall/query-percentage')
    ax = sns.heatmap(df, linewidths=.5, annot=True)
    plt.show()


def right2left():
	"""
	Returns: Right wrt left = TL_R = TR2L

	LHS -> Ry(-90) -> Rx(90) -> RHS

	TL_R = Ry(-90) Rx(90)
	
	"""


	thetaX = np.radians(90)
	thetaY = np.radians(-90)

	Rx = np.array([[1, 0, 0], [0, np.cos(thetaX), -np.sin(thetaX)], [0, np.sin(thetaX), np.cos(thetaX)]])
	Ry = np.array([[np.cos(thetaY), 0, np.sin(thetaY)], [0, 1, 0], [-np.sin(thetaY), 0, np.cos(thetaY)]])

	TR2L = np.identity(4)
	TR2L[0:3, 0:3] =  Ry @ Rx 
	
	return TR2L



def L2R(TC1C2L):
	TLR = right2left()
	TC1C2R = np.linalg.inv(TLR) @ TC1C2L @ TLR
	# Camera wrt Base in Right hand frame
	TBC = np.eye(4)
	# TBC[1,3]+=1
	# B2 wrt B1 in Right hand frame
	TB1B2R = TBC @ TC1C2R @ np.linalg.inv(TBC)
	return TB1B2R


if __name__ == "__main__":
    opt = parser.parse_args()
    dataset = get_whole_val_set(opt.root_dir, opt.dataset.lower())

    db_lst = dataset.dbStruct.dbImage
    db_lst = np.array([int(re.search('\d+', x.replace(' ',''))[0]) for x in db_lst])
    q_lst = dataset.dbStruct.qImage
    q_lst = np.array([int(re.search('\d+', x.replace(' ',''))[0]) for x in q_lst])
    db_cood_lst = dataset.dbStruct.locDb
    q_cood_lst = dataset.dbStruct.locQ
    # X_Db, Y_Db = db_cood_lst[:,0], db_cood_lst[:,1]
    # X_Q, Y_Q = q_cood_lst[:,0], q_cood_lst[:,1]

    # since we didn't included Rot. in mat files that's why we need to do this.

    df = pd.read_csv(opt.reference_poses)
    col = df.columns
    # print(df[col[1:4]])
    db_lst = np.array(db_lst)-1
    [ db_qw, db_qx, db_qy, db_qz] = [df[col[4]][db_lst], df[col[5]][db_lst], df[col[6]][db_lst], df[col[7]][db_lst]]
    quat_Db = np.vstack((db_qx, db_qy,db_qz, db_qw)).T
    Rot_Db = R.from_quat(quat_Db).as_matrix()
    Transform_Db = np.zeros((Rot_Db.shape[0],4,4))
    Transform_Db[:,:3,:3] = Rot_Db
    Transform_Db[:,0,3] = df[col[1]][db_lst]
    Transform_Db[:,1,3] = df[col[2]][db_lst] + 1
    Transform_Db[:,2,3] = df[col[3]][db_lst] 
    Transform_Db[:,3,3] = np.ones(Rot_Db.shape[0])
    print(len(Transform_Db))

    df = pd.read_csv(opt.query_poses)
    col = df.columns
    # print(df[col[1:4]])

    q_lst = np.array(q_lst)-1
    #  uncomment for small_2
    # q_lst = q_lst[::2]
    [ q_qw, q_qx, q_qy, q_qz] = [df[col[4]][q_lst], df[col[5]][q_lst], df[col[6]][q_lst], df[col[7]][q_lst]]
    quat_Q = np.vstack((q_qx, q_qy,q_qz, q_qw)).T
    Rot_Q = R.from_quat(quat_Q).as_matrix()
    Transform_Q = np.zeros((Rot_Q.shape[0],4,4))
    Transform_Q[:,:3,:3] = Rot_Q
    Transform_Q[:,0,3] = df[col[1]][q_lst]
    Transform_Q[:,1,3] = df[col[2]][q_lst] + 1
    Transform_Q[:,2,3] = df[col[3]][q_lst] 
    Transform_Q[:,3,3] = np.ones(Rot_Q.shape[0])
    print(len(Transform_Q))



    RM = open(opt.rord_match_count, "r")
    RML = RM.readlines()
    RM.close()
    best_dict = {}
    for l in RML:
        l = l.rstrip()
        (_, rgb1, _, rgb2, count) = l.split()
        count = int(count)
        idx1 = int(re.findall('\d+', rgb1)[0]) - 1
        idx2 = int(re.findall('\d+', rgb2)[0]) - 1
        #uncomment for small_2
        # idx1 = int(re.findall('\d+', rgb1)[2]) - 1
        # idx2 = int(re.findall('\d+', rgb2)[2]) - 1
        # print(idx1, idx2)
        if idx1 in best_dict.keys():
            [rid, ct] = best_dict[idx1]
            if ct<count:
                best_dict[idx1]=[idx2, count]
        else:
            best_dict[idx1]=[idx2, count]



    data = []
    preds = []
    truths = []
    query = []
    retrieved = []
    for idx1 in best_dict.keys():
        [idx2,_] = best_dict[idx1]
        query.append(idx1)
        retrieved.append(idx2)
        u = np.argwhere(q_lst==idx1)[0][0]
        v = np.argwhere(db_lst==idx2)[0][0]
        R_gt = np.linalg.inv(Transform_Db[v]) @ Transform_Q[u]
        R_pred = np.load(os.path.join(opt.rord_trans,str(idx1+1)+"-"+str(1+idx2)+".npy"))
        # R_pred = L2R(R_pred)
        # np.set_printoptions(precision=1)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

        # print("query idx = ", idx1+1, ' retrieved idx = ',idx2+1, "\n\nTransform_Q[u] = \n\n", Transform_Q[u], "\n\nTransform_Db[v] = \n\n", Transform_Db[v], "\n\nR_gt = \n\n", R_gt, "\n\n\nR_pred = \n\n", R_pred, "\n\n\n")
        # R_pred[2,3] = 0
        truths.append(R_gt) 
        preds.append(R_pred) 



    # netvlad_candidates = np.load(opt.netvlad_predictions)
    # for i in range(netvlad_candidates.shape[0]):
    #     for j in range(1):
    #         u = np.argwhere(q_lst==q_lst[i])[0][0]
    #         v = np.argwhere(db_lst==db_lst[int(netvlad_candidates[i,j])])[0][0]
    #         R_gt = np.linalg.inv(Transform_Q[u]) @ Transform_Db[v]
    #         R_pred = np.load(os.path.join(opt.rord_trans,str(q_lst[i]+1)+"-"+str(1+db_lst[int(netvlad_candidates[i,j])])+".npy"))
    #         # R_pred[2,3] = 0
    #         truths.append(R_gt) 
    #         preds.append(R_pred) 
    #         if(i==0):
    #             print(R_gt,  "\n\n\n")
    #             print(R_pred)

    rot_errors, trans_errors = get_errors(preds, truths, query, retrieved)

    
    data.append({
        'name': 'Query-Retrieved ('+ opt.name + ")",
        'rot_errors': rot_errors,
        'trans_errors': trans_errors,
    })
    print_statistics(data)   
