from typing import Callable, Dict, List, Tuple, Union
import cv2
import numpy as np
import quaternion
from sys import argv
import pandas as pd
from scipy.spatial.transform import Rotation as R
from PIL import Image
import pydegensac
import open3d as o3d
import torch

DEPTH_SCALE = 65535 / 10



def normalize_pixel_coords(coords: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Transform pixel coordinates from (0,W) to (-1,1) and (0,H) to (1,-1)
    """
    coords = 2 * coords / np.array(shape) - 1
    coords[:, 1] *= -1  # indexing array Y is top-down, whereas world Y is bottom-up
    return coords


def denormalize_pixel_coords(coords: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Invert the transformation performed by `normalize_pixel_coords` function
    """
    coords[:, 1] *= -1
    return 0.5 * np.array(shape) * (coords + 1)
    
def pose_to_Rt(pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # t = pose[:3] + [0, 1.5, 0]  # getting sensor pose from agent pose
        t = pose[:,3]
        R = quaternion.from_float_array(pose[3:])
        R = quaternion.as_rotation_matrix(R)
        return R, t.reshape(-1, 1)



def filter_matches(xy1: np.ndarray, xy2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter matching keypoints by estimating a mapping (homography or fundamental
    matrix) that best relates the tranformation between the image pair
    """
    global FIND_FUNDAMENTAL_MATRIX, FIND_HOMOGRAPHY, RANSAC_CONFIG

    # TODO check if opencv RANSAC works as well as pydegensac
    if FIND_FUNDAMENTAL_MATRIX:
        return pydegensac.findFundamentalMatrix(xy1, xy2, **RANSAC_CONFIG["F"])
        # _, homo_idx = cv2.findFundamentalMat(
        #     xy1, xy2, cv2.USAC_DEFAULT, 10.0, 0.99, 10000
        # )
    elif FIND_HOMOGRAPHY:
        return pydegensac.findHomography(xy1, xy2, **RANSAC_CONFIG["H"])
        # _, homo_idx = cv2.findHomography(
        #     xy1, xy2, cv2.USAC_DEFAULT, 10.0, None, 10000, 0.99
        # )
    else:
        return np.array([]), np.ones(xy1.shape[0], dtype=bool)


def estimate(
    M: np.ndarray, K: np.ndarray, keypoints1: np.ndarray, keypoints2: np.ndarray
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes a Fundamental matrix, a camera intrinsics matrix, a pair of corresponding
    keypoints to output the R and t transform between the 2 images
    """
    E = K.T @ M @ K
    # TODO try tuning other parameters for normalized coords
    ret, R, t, mask = cv2.recoverPose(E, keypoints1, keypoints2, K)
    T = np.hstack((R, t))
    mask = np.bool8(mask.reshape(-1))
    keypoints1 = keypoints1[mask]
    keypoints2 = keypoints2[mask]
    return ret, T, keypoints1, keypoints2


def get_GT_matches(
    keypoints1: np.ndarray,
    depthpath1: str,
    k: np.ndarray,
    pose_pair: Tuple[np.ndarray, np.ndarray],
):
    """
    Given keypoints from one color image, the corresponding depth values, camera
    intrinsics matrix and a pair of ground truth camera poses, this function calculates
    the ground truth correspondences in the second image
    Arguments :
        `keypoints1` : keypoints to reproject onto the second image
        `imgpath1` : path to the first image file (to get the depth file)
        `K` : camera intrinsics matrix
        `pose_pair` : pair of ground truth camera poses of 2 images
    """
    global DEPTH_SCALE
    T1 = np.eye(4)
    T2 = np.eye(4)
    K = np.eye(4)

    T1 = pose_pair[0]
    T2 = pose_pair[1]
    # K[:3, :3] = k

    # get depth values for each keypoint
    depth = np.float64(cv2.imread(depthpath1, cv2.IMREAD_ANYDEPTH)) / DEPTH_SCALE 
    kps = np.uint16(keypoints1.round(0))
    kp_depths = depth[kps[:, 1], kps[:, 0]].reshape(-1, 1).T
    # print("depth = ",depth)
    # print("kps = ",kps)
    # print("kp_depths = ",kp_depths)

    # transform keypoints from one camera to the other
    mask = ~np.isclose(kp_depths, 0).squeeze()  # mask out all zero depths
    keypoints1 = normalize_pixel_coords(keypoints1, depth.shape)
    xys1 = np.ones((4, kp_depths.size))
    xys1[:2, :] = keypoints1.T * kp_depths
    xys1[2, :] = -kp_depths
    xys2 = K @ np.linalg.inv(T2) @ T1 @ np.linalg.inv(K) @ xys1

    # print("mask = ",mask)

    # mask out non-negative Z values and then divide to get normalized pixel indices
    mask *= xys2[2] < 0

    xys2 = -np.true_divide(xys2[:2, :], xys2[2, :], where=(xys2[2] != 0))
    # mask out all coordinates that are outside (-1,1)
    mask *= np.prod((xys2 < 1) * (xys2 >= -1), axis=0, dtype=bool)
    print(mask)
    return denormalize_pixel_coords(xys2.T[mask], depth.shape), mask

def readDepth(depthFile):
	depth = Image.open(depthFile)
	if depth.mode != "I":
		raise Exception("Depth image is not in intensity format")

	return (np.asarray(depth)*10)/65535


def getPointCloud(keypoints, depthFile):
	thresh = 15.0
	# thresh = 5.6


	depth = readDepth(depthFile)

	points = []
	scalingFactor, centerX, centerY, focalX, focalY = -1, 400, 400, -400, 400
    

	for [v,u] in keypoints:
			v, u = int(v), int(u)
			Z = depth[v, u] / scalingFactor
			if Z==0: continue

			X = (u - centerX) * Z / focalX
			Y = (v - centerY) * Z / focalY

			points.append((X, Y, Z))

	points = np.asarray(points)
	points = points.T
	points /= points[2,:]
	return  points



def getSphere(pts):
	sphs = []

	for ele in pts:
		if(ele is not None):
			sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
			sphere.paint_uniform_color([0.9, 0.2, 0])

			trans = np.identity(4)
			trans[0, 3] = ele[0]
			trans[1, 3] = ele[1]
			trans[2, 3] = ele[2]

			sphere.transform(trans)
			sphs.append(sphere)

	return sphs


def preprocess_image(image, preprocessing=None):
    image = image.astype(np.float32)
    image = np.transpose(image, [2, 0, 1])
    if preprocessing is None:
        pass
    elif preprocessing == 'caffe':
        # RGB -> BGR
        image = image[:: -1, :, :]
        # Zero-center by mean pixel
        mean = np.array([103.939, 116.779, 123.68])
        image = image - mean.reshape([3, 1, 1])
    elif preprocessing == 'torch':
        image /= 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean.reshape([3, 1, 1])) / std.reshape([3, 1, 1])
    else:
        raise ValueError('Unknown preprocessing parameter.')
    return image


def read_and_process_image(img_path, resize=None, H=None, h=None, w=None, preprocessing='caffe'):
	img1 = Image.open(img_path)
	if resize:
		img1 = img1.resize(resize)
	if(img1.mode != 'RGB'):
		img1 = img1.convert('RGB')
	img1 = np.array(img1)
	if H is not None:
		img1 = cv2.warpPerspective(img1, H, dsize=(400, 400))
		# cv2.imshow("Image", cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
		# cv2.waitKey(0)
	igp1 = torch.from_numpy(preprocess_image(img1, preprocessing=preprocessing).astype(np.float32))
	return igp1, img1






def drawOrg(image1, image2, orgSrc, orgDst):
	img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
	img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

	for i in range(orgSrc.shape[1]):
		im1 = cv2.circle(img1, (int(orgSrc[0, i]), int(orgSrc[1, i])), 3, (0, 0, 255), 1)
	for i in range(orgDst.shape[1]):
		im2 = cv2.circle(img2, (int(orgDst[0, i]), int(orgDst[1, i])), 3, (0, 0, 255), 1)

	im4 = cv2.hconcat([im1, im2])
	for i in range(orgSrc.shape[1]):
		im4 = cv2.line(im4, (int(orgSrc[0, i]), int(orgSrc[1, i])), (int(orgDst[0, i]) +  im1.shape[1], int(orgDst[1, i])), (0, 255, 0), 1)
	im4 = cv2.cvtColor(im4, cv2.COLOR_BGR2RGB)
	# cv2.imshow("Image", im4)
	# cv2.waitKey(0)

	return im4


if __name__=="__main__":
    keypoints_gt = np.load(argv[1])
    # keypoints_gt = keypoints_gt[:,:,:10]
    keypoints1 = []
    keypoints1.append(keypoints_gt[0].T)
    keypoints1.append(keypoints_gt[1].T)
    keypoints1 = np.array(keypoints1)
    depthpath1 = argv[2]
    reference_poses = argv[3]
    query_poses = argv[4]
    rgbpath1 = argv[5]
    rgbpath2 = argv[6]



    df = pd.read_csv(reference_poses)
    col = df.columns
    # db_lst = 202-1
    # db_lst = 712-1
    # db_lst = 502-1
    db_lst = 178-1
    # db_lst = 520-1
    # db_lst = 127-1
    # db_lst = 139-1
    [ db_qw, db_qx, db_qy, db_qz] = [df[col[4]][db_lst], df[col[5]][db_lst], df[col[6]][db_lst], df[col[7]][db_lst]]
    quat_Db = np.vstack((db_qx, db_qy,db_qz, db_qw)).T
    Rot_Db = R.from_quat(quat_Db).as_matrix()
    Transform_Db = np.zeros((Rot_Db.shape[0],4,4))
    Transform_Db[:,:3,:3] = Rot_Db
    Transform_Db[:,0,3] =  df[col[1]][db_lst]
    Transform_Db[:,1,3] = df[col[2]][db_lst] 
    Transform_Db[:,2,3] = df[col[3]][db_lst] + 1
    Transform_Db[:,3,3] = np.ones(Rot_Db.shape[0])


    df = pd.read_csv(query_poses)
    col = df.columns
    # q_lst = 9-1
    # q_lst = 1-1
    # q_lst = 49-1
    q_lst = 65-1
    # q_lst = 21-1
    # q_lst = 193-1
    # q_lst = 113-1
    [ q_qw, q_qx, q_qy, q_qz] = [df[col[4]][q_lst], df[col[5]][q_lst], df[col[6]][q_lst], df[col[7]][q_lst]]
    quat_Q = np.vstack((q_qx, q_qy,q_qz, q_qw)).T
    Rot_Q = R.from_quat(quat_Q).as_matrix()
    Transform_Q = np.zeros((Rot_Q.shape[0],4,4))
    Transform_Q[:,:3,:3] = Rot_Q
    Transform_Q[:,0,3] = df[col[1]][q_lst]
    Transform_Q[:,1,3] = df[col[2]][q_lst]
    Transform_Q[:,2,3] = df[col[3]][q_lst] + 1
    Transform_Q[:,3,3] = np.ones(Rot_Q.shape[0])

    Transform_Q = Transform_Q[0]
    Transform_Db = Transform_Db[0]


    K = np.array([
        [-400, 0, 400],
        [0, 400, 400],
        [0, 0, 1],
    ])
    # gt2, gt_mask = get_GT_matches(keypoints1[0], depthpath1, np.eye(3), [Transform_Q,Transform_Db])
#     # print(keypoints1[0].shape,kp.shape)
    depth = readDepth(depthpath1)
    # points = getPointCloud(keypoints1[0], depthpath1)  # 3*N
    Homogenous_kp1 = np.hstack((keypoints1[0],np.ones((keypoints1[0].shape[0],1))))
    KP1 = []
    for [v, u] in keypoints1[0]:
        lamda = depth[int(u), int(v)]/(-1)
        KP1.append(lamda*np.array([v,u,1]))
    Homogenous_kp1 = np.array(KP1).T

    X_qc = np.linalg.inv(K) @ Homogenous_kp1 # camera frame
    # o3d.visualization.draw_geometries(getSphere(X_qc.T))

    X_qc = Transform_Q[:3,:3]  @ X_qc # world frame
    X_w = (X_qc.T + Transform_Q[:3,3].reshape((1,3)))
    # o3d.visualization.draw_geometries(getSphere(X_w))

    X_dbc = (X_w - Transform_Db[:3,3].reshape((1,3))).T
    kp =  np.linalg.inv(Transform_Db[:3,:3]) @ X_dbc
    # o3d.visualization.draw_geometries(getSphere(kp.T))

    kp = K @ kp
    kp = kp/kp[2,:]
    kp = kp.T
    kp = kp[:,:2]
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
    # for i in range(len(kp)):
    #     print(kp[i,:2], " and ", keypoints1[1][i])

    print(keypoints_gt[0].shape, kp.T[:2].shape)

#     _, img1 = read_and_process_image(rgbpath1)
#     _, img2 = read_and_process_image(rgbpath2)
#     # image3=drawOrg(img1, img2, keypoints_gt[0], keypoints_gt[1])
#     # cv2.imshow('Measure view', image3)
#     # cv2.waitKey()


#     # image3=drawOrg(img1, img2, keypoints_gt[0], kp.T[:2])
#     # cv2.imshow('GT view', image3)
#     # cv2.waitKey()

# # (4, 2) (63,)
    gt_mask = np.prod((kp > 0) * (kp < 800), axis=1, dtype=bool)
    gt2 = kp[gt_mask]

#     print(gt2.shape, gt_mask.shape)


    image1 = cv2.cvtColor(cv2.imread(rgbpath1), cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(cv2.imread(rgbpath2), cv2.COLOR_BGR2RGB)

    matches_image = cv2.drawMatches(
        image1,
        [cv2.KeyPoint(*xy, 1) for xy in keypoints1[0][~gt_mask]],
        image2,
        [cv2.KeyPoint(*xy, 1) for xy in keypoints1[1][~gt_mask]],
        [cv2.DMatch(idx, idx, 1) for idx in range(np.sum(~gt_mask))],
        None,
        matchColor=(0, 0, 150),
    )
    # cv2.imshow('Non-gt avail view', matches_image)
    # cv2.waitKey()

    # drawing only the matches with valid ground truth correspondences
    cv2.drawMatches(
        image1,
        [cv2.KeyPoint(*xy, 1) for xy in keypoints1[0][gt_mask]],
        image2,
        [cv2.KeyPoint(*xy, 1) for xy in keypoints1[1][gt_mask]],
        [cv2.DMatch(idx, idx, 1) for idx in range(np.sum(gt_mask))],
        matches_image,
        matchColor=(0, 250, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,
    )

    # nothing to draw in case of no GT pose
    # get euclidean distance between every matched and reprojected feature pair
    keypts1 = keypoints1[1][gt_mask]  # we need only matches with ground truth
    print("keypts1 = ",keypts1)
    print("gt2 = ",gt2)
    gt_xy_diff = np.sqrt(np.sum((gt2 - keypts1) ** 2, axis=1))
    gt_xy_diff = np.uint16(gt_xy_diff.round(0))
    keypts1 = np.uint16(keypts1.round(0)) + [image1.shape[1], 0]
    gt2 = np.uint16(gt2.round(0)) + [image1.shape[1], 0]
    for i in range(np.sum(gt_mask)):
        # draw a circle centered at matched feature with radius being the error distance
        matches_image = cv2.circle(
            matches_image,
            keypts1[i],
            gt_xy_diff[i],
            (255, 0, 0),
            1,
        )
        # draw a line connecting matched and reprojected features
        matches_image = cv2.line(matches_image, keypts1[i], gt2[i], (255, 0, 0), 1)

    cv2.imshow('GT avail view', matches_image)
    cv2.waitKey()