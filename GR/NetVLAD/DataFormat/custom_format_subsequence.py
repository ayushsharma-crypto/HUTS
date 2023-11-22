import os
import numpy as np
import pandas as pd
from sys import argv

ROOT = argv[1]
COLOR = os.path.join(ROOT, argv[2])
POSES = os.path.join(ROOT, argv[3])
SAVE_POSES = os.path.join(ROOT, argv[4]+"_poses")



df = pd.read_csv(POSES)
col = df.columns
pose = []

def ks(x):
    return int(x[:-4])

files = sorted(os.listdir(COLOR), key=ks)
for f in files:
    i = int(f[:-4])
    print(i)
    pose.append([df[col[1]][i], df[col[3]][i]])

pose = np.array(pose)
np.save(SAVE_POSES,pose)