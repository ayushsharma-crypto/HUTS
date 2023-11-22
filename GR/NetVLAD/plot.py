from cProfile import label
from pathlib import Path
import argparse
import os
from pickle import NONE
import numpy as np
import pandas as pd
from NetVLAD.dataset import get_whole_val_set
import re
import matplotlib.pyplot as plt
import networkx as nx
import chart_studio.plotly as py
import plotly.graph_objs as go
import seaborn as sns

parser = argparse.ArgumentParser(description='PLOTTING THE RETRIEVAL')
parser.add_argument('--root_dir', type=str, default='', help='Path to dataset')
parser.add_argument('--dataset', type=str, default='berlin', 
        help='Dataset to use', choices=['small_2','spair_90', 'spair_180','SMALL','TOPVIEW','PERSPECTIVE','sT4fr6TAbpF','oxford', 'nordland', 'berlin'])
parser.add_argument('--netvlad_predictions', type=str, default='', help='Path to NetVLAD Predictions')
parser.add_argument('--plot_names', type=str, default='', help='Plot names')




def plot_3d_class_all(X_Db, Y_Db, X_Q, Y_Q,  db_lst, q_lst, plot_names, thr=3):
    q_lst = np.array(q_lst)
    db_lst = np.array(db_lst)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X_Db, Y_Db, [1 for _ in range(len(X_Db))], c='b')
    ax.scatter3D(X_Q[:7], Y_Q[:7], [0 for _ in range(7)], c='g')
    ax.scatter3D(X_Q[7:20], Y_Q[7:20], [0 for _ in range(7,20)], c='r')
    ax.scatter3D(X_Q[20:48], Y_Q[20:48], [0 for _ in range(20,48)], c='orange')
    ax.scatter3D(X_Q[48:56], Y_Q[48:56], [0 for _ in range(48,56)], c='r')
    ax.scatter3D(X_Q[56:69], Y_Q[56:69], [0 for _ in range(56,69)], c='g')
    ax.scatter3D(X_Q[69:80], Y_Q[69:80], [0 for _ in range(69,80)], c='orange')
    for i in range(len(X_Db)-1):
        plt.plot([X_Db[i], X_Db[i+1]],[Y_Db[i],Y_Db[i+1]],[1,1],'b-')
    for i in range(7):
        plt.plot([X_Q[i], X_Q[i+1]],[Y_Q[i],Y_Q[i+1]],[0,0],'g-')
    for i in range(7,20):
        plt.plot([X_Q[i], X_Q[i+1]],[Y_Q[i],Y_Q[i+1]],[0,0],'r-')
    for i in range(20,48):
        plt.plot([X_Q[i], X_Q[i+1]],[Y_Q[i],Y_Q[i+1]],[0,0],'y-')
    for i in range(48,56):
        plt.plot([X_Q[i], X_Q[i+1]],[Y_Q[i],Y_Q[i+1]],[0,0],'r-')
    for i in range(56,69):
        plt.plot([X_Q[i], X_Q[i+1]],[Y_Q[i],Y_Q[i+1]],[0,0],'g-')
    for i in range(69,79):
        plt.plot([X_Q[i], X_Q[i+1]],[Y_Q[i],Y_Q[i+1]],[0,0],'y-')
    
    
    ax.set_zlabel('Z-Axis')
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_aspect('auto', 'datalim')
    plt.legend(["Reference", "Query Points have same side view", "Query without any views","Query Points having some view"], loc="upper right")
    plt.title("Retrieval for views without having SSV in reference images but some view available with visual overlap")
    plt.show()





def plot_3d_class(X_Db, Y_Db, X_Q, Y_Q, netvlad_candidates, db_lst, q_lst, plot_names, thr=7.8):
    q_lst = np.array(q_lst)
    db_lst = np.array(db_lst)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X_Db, Y_Db, [1 for _ in range(len(X_Db))], c='b')
    ax.scatter3D(X_Q, Y_Q, [0 for _ in range(len(X_Q))], c='orange')
    for i in range(len(X_Db)-1):
        plt.plot([X_Db[i], X_Db[i+1]],[Y_Db[i],Y_Db[i+1]],[1,1],'b-')
    for i in range(len(X_Q)-1):
        plt.plot([X_Q[i], X_Q[i+1]],[Y_Q[i],Y_Q[i+1]],[0,0],'y-')
    
    # count=0
    # for i in range(45,netvlad_candidates.shape[0]):
    #     for j in range(1):
    #         u = np.argwhere(q_lst==q_lst[i])[0][0]
    #         v = np.argwhere(db_lst==db_lst[int(netvlad_candidates[i,j])])[0][0]
    #         p1 = np.array([X_Q[u], Y_Q[u]])
    #         p2 = np.array([X_Db[v], Y_Db[v]])
    #         if np.linalg.norm(p1-p2)>= thr:
    #             count+=1
    #             print(q_lst[i],db_lst[v])
    #             plt.plot([X_Q[u], X_Db[v]],[Y_Q[u], Y_Db[v]],[0,1],"k-")
    #         else:
    #             print("Inside = ",q_lst[i], db_lst[v])
    # print(count)
    ax.set_zlabel('Z-Axis')
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_aspect('auto', 'datalim')
    plt.legend(["Reference", "Query"], loc="upper right")
    plt.title("Retrieval for views without having SSV in reference images but some view available with visual overlap")
    plt.show()




def plot_3d(X_Db, Y_Db, X_Q, Y_Q, netvlad_candidates, db_lst, q_lst, plot_names, thr=3):
    q_lst = np.array(q_lst)
    db_lst = np.array(db_lst)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X_Db, Y_Db, [0 for _ in range(len(X_Db))], cmap='Green',label="Database Points")
    ax.scatter3D(X_Q, Y_Q, [1 for _ in range(len(X_Q))], cmap='Red',label="Query Points")
    for i in range(len(X_Db)-1):
        plt.plot([X_Db[i], X_Db[i+1]],[Y_Db[i],Y_Db[i+1]],[0,0],'b-')
    for i in range(len(X_Q)-1):
        plt.plot([X_Q[i], X_Q[i+1]],[Y_Q[i],Y_Q[i+1]],[1,1],'r-')
    
    count=0

    for i in range(netvlad_candidates.shape[0]):
        for j in range(1):
            u = np.argwhere(q_lst==q_lst[i])[0][0]
            v = np.argwhere(db_lst==db_lst[int(netvlad_candidates[i,j])])[0][0]
            p1 = np.array([X_Q[u], Y_Q[u]])
            p2 = np.array([X_Db[v], Y_Db[v]])
            if np.linalg.norm(p1-p2)>= thr:
                count+=1
                plt.plot([X_Q[u], X_Db[v]],[Y_Q[u], Y_Db[v]],[1,0],"k-")
    print("count = ", count)
    ax.set_zlabel('Z-Axis')
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_aspect('auto', 'datalim')
    plt.legend(["Reference","Query"], loc="upper right")
    plt.title(plot_names+": Query and Reference Imgs path, with connection \n having Translation Frob. norm threshold >= "+str(thr))
    plt.show()



def plot_3d_opposite(X_Db, Y_Db, X_Q, Y_Q, netvlad_candidates, db_lst, q_lst, plot_names, thr=3):
    q_lst = np.array(q_lst)
    db_lst = np.array(db_lst)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X_Db, Y_Db, [0 for _ in range(len(X_Db))], cmap='Green',label="Database Points")
    ax.scatter3D(X_Q, Y_Q, [1 for _ in range(len(X_Q))], cmap='Red',label="Query Points")
    for i in range(len(X_Db)-1):
        plt.plot([X_Db[i], X_Db[i+1]],[Y_Db[i],Y_Db[i+1]],[0,0],'b-')
    for i in range(len(X_Q)-1):
        plt.plot([X_Q[i], X_Q[i+1]],[Y_Q[i],Y_Q[i+1]],[1,1],'r-')
    
    count=0

    for i in range(netvlad_candidates.shape[0]):
        for j in range(1):
            u = np.argwhere(q_lst==q_lst[i])[0][0]
            v = np.argwhere(db_lst==db_lst[int(netvlad_candidates[i,j])])[0][0]
            p1 = np.array([X_Q[u], Y_Q[u]])
            p2 = np.array([X_Db[v], Y_Db[v]])
            if np.linalg.norm(p1-p2)< thr:
                count+=1
                plt.plot([X_Q[u], X_Db[v]],[Y_Q[u], Y_Db[v]],[1,0],"k-")
    print("count = ", count)
    ax.set_zlabel('Z-Axis')
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_aspect('auto', 'datalim')
    plt.legend(["Reference","Query"], loc="upper right")
    plt.title(plot_names+": Query and Reference Imgs path, with connection \n having Translation Frob. norm threshold < "+str(thr))
    plt.show()


def plot_line(X_Db, Y_Db, X_Q, Y_Q, netvlad_candidates, db_lst, q_lst, plot_names):
    q_lst = np.array(q_lst)
    db_lst = np.array(db_lst)
    W, H = netvlad_candidates.shape
    H = 2
    s = 0
    LABL = [ str(i+1)+"-best" for i in range(s,H)]
    NM = np.empty((W,H))
    for i in range(W):
        for j in range(s,H):
            u = np.argwhere(q_lst==q_lst[i])[0][0]
            v = np.argwhere(db_lst==db_lst[int(netvlad_candidates[i,j])])[0][0]
            p1 = np.array([X_Q[u], Y_Q[u]])
            p2 = np.array([X_Db[v], Y_Db[v]])
            NM[i,j] = np.linalg.norm(p1-p2)

    for h in range(s,H):
        plt.plot(range(len(NM)),NM[:,h])
    plt.xlabel("Query IDX")
    plt.ylabel("Geological pos. Norm")
    plt.legend(LABL, loc="upper right")
    plt.yticks(np.arange(0, 30, step=2))
    plt.xticks(np.arange(0, 100, step=10))
    plt.title(plot_names+": Frobenius Norm on query & retrieved image geo-location Vs Query Index")
    plt.show()



def plot_histogram(X_Db, Y_Db, X_Q, Y_Q, netvlad_candidates, db_lst, q_lst, plot_names):
    q_lst = np.array(q_lst)
    db_lst = np.array(db_lst)
    W, H = netvlad_candidates.shape
    H = 2
    s = 0
    LABL = [ str(i+1)+"-best" for i in range(s,H)]
    NM = np.empty((W,H))
    for i in range(W):
        for j in range(s,H):
            u = np.argwhere(q_lst==q_lst[i])[0][0]
            v = np.argwhere(db_lst==db_lst[int(netvlad_candidates[i,j])])[0][0]
            p1 = np.array([X_Q[u], Y_Q[u]])
            p2 = np.array([X_Db[v], Y_Db[v]])
            NM[i,j] = np.linalg.norm(p1-p2)

    sns.set(style="darkgrid")
    DF = pd.DataFrame({'norm':[], 'color':[]})
    for h in range(s,H):
        data = {'norm':NM[:,h], 'color':[str(h)+"-best"]*W}
        df = pd.DataFrame(data)
        DF = DF.append(df, ignore_index=True)
    sns.histplot(data = DF,binwidth=0.25, x="norm",hue="color",kde = True)

    plt.legend(LABL, loc="upper right") 
    plt.ylabel("Query IDX count")
    plt.title(plot_names+": Histogram for number of query images in given norm range")
    plt.show()



if __name__ == "__main__":
    opt = parser.parse_args()
    dataset = get_whole_val_set(opt.root_dir, opt.dataset.lower())

    db_lst = dataset.dbStruct.dbImage
    db_lst = [int(re.search('\d+', x.replace(' ',''))[0]) for x in db_lst]
    q_lst = dataset.dbStruct.qImage
    q_lst = [int(re.search('\d+', x.replace(' ',''))[0]) for x in q_lst]
    db_cood_lst = dataset.dbStruct.locDb
    q_cood_lst = dataset.dbStruct.locQ
    X_Db, Y_Db = db_cood_lst[:,0], db_cood_lst[:,1]
    X_Q, Y_Q = q_cood_lst[:,0], q_cood_lst[:,1]

    netvlad_candidates = np.load(opt.netvlad_predictions)

    # uncomment for small_2
    X_Q = X_Q[::2]
    Y_Q = Y_Q[::2]

    plot_3d_class_all(X_Db, Y_Db, X_Q, Y_Q, db_lst, q_lst, opt.plot_names)

    # plot_3d(X_Db, Y_Db, X_Q, Y_Q, netvlad_candidates, db_lst, q_lst, opt.plot_names)
    # plot_3d_opposite(X_Db, Y_Db, X_Q, Y_Q, netvlad_candidates, db_lst, q_lst, opt.plot_names)
    # plot_line(X_Db, Y_Db, X_Q, Y_Q, netvlad_candidates, db_lst, q_lst, opt.plot_names)
    # plot_histogram(X_Db, Y_Db, X_Q, Y_Q, netvlad_candidates, db_lst, q_lst, opt.plot_names)

