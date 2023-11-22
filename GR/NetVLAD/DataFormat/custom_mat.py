from scipy.io import savemat
import os
import numpy as np
import pandas as pd
from sys import argv

data_path = argv[1]
save_mat_name = os.path.join(data_path,argv[2]+".mat")
d_path = os.path.join(data_path,"Reference_resized")
q_path = os.path.join(data_path,"Query_resized")
d_cood_path = os.path.join(data_path,"Reference_resized_poses.npy")
q_cood_path = os.path.join(data_path,"Query_resized_poses.npy")

def ks(x):
    return int(x[:-4])
d_list = np.array( sorted(os.listdir(d_path), key=ks))
q_list = np.array( sorted(os.listdir(q_path), key=ks))
d_img = ["Reference_resized/"+fname for fname in d_list]
q_img = ["Query_resized/"+fname for fname in q_list]

d_cood = np.load(d_cood_path)
q_cood = np.load(q_cood_path)
    
mat_data = {
    'dbStruct' : []
}
mat_data['dbStruct'].append(d_img)
mat_data['dbStruct'].append(d_cood)
mat_data['dbStruct'].append(q_img)
mat_data['dbStruct'].append(q_cood)
mat_data['dbStruct'].append(np.array([np.array([len(d_img)])]))
mat_data['dbStruct'].append(np.array([np.array([len(q_img)])]))
mat_data['dbStruct'].append(np.array([np.array([2])]))
mat_data['dbStruct'].append(np.array([np.array([4])]))
savemat(save_mat_name, mat_data)