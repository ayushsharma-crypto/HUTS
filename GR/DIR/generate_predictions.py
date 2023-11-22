import numpy as np
import faiss
from sys import argv
import os
import pandas as pd

if __name__=="__main__":
    dbFeat = np.load(argv[1])
    qFeat = np.load(argv[2])
    print('====> Building faiss index with pool_size = ', qFeat.shape[1])
    faiss_index = faiss.IndexFlatL2(qFeat.shape[1])
    faiss_index.add(dbFeat)
    n_values = [1,5,10,20]
    distances, predictions = faiss_index.search(qFeat, max(n_values))
    save_path = os.path.dirname(argv[1])
    np.save(os.path.join(save_path, 'dir_preds.npy'), predictions)
    np.save(os.path.join(save_path, 'dir_dist.npy'), distances)


    dir_candidates = np.load(os.path.join(save_path, 'dir_preds.npy'))
    dir_distances = np.load(os.path.join(save_path, 'dir_dist.npy'))
    print("dir_candidates= ",dir_candidates)
    print("dir_distances= ",dir_distances)

    q_lst_file = open(os.path.join(save_path, 'q_list.txt'))
    db_lst_file = open(os.path.join(save_path, 'db_list.txt'))
    q_lst = q_lst_file.readlines()
    db_lst = db_lst_file.readlines()
    q_lst_file.close()
    db_lst_file.close()

    candidate_list = []
    candidate_dist_list = []
    for i in range(dir_candidates.shape[0]):
      for j in range(dir_candidates.shape[1]):
        Q = q_lst[i].strip()
        DB = db_lst[dir_candidates[i,j]].strip()
        candidate_list.append([Q, DB])
        candidate_dist_list.append([Q, DB, dir_distances[i,j]])

    df = pd.DataFrame.from_records(candidate_list)
    df.to_csv(os.path.join(save_path, 'dir_candidate_list.txt'), header=None, index=None, sep=' ', mode='a')
    df = pd.DataFrame.from_records(candidate_dist_list)
    df.to_csv(os.path.join(save_path, 'dir_candidate_distance_list.txt'), header=None, index=None, sep=' ', mode='a')