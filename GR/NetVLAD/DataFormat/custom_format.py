import os
import numpy as np
import pandas as pd
from sys import argv
from PIL import  Image

FREQ=int(argv[1])
ROOT = argv[2]
SAVE_ROOT = argv[3]

COLOR = os.path.join(ROOT, 'color')
DEPTH = os.path.join(ROOT, 'depth')
POSES = os.path.join(ROOT, 'poses.csv')

SAVE_COLOR = os.path.join(SAVE_ROOT, argv[4])
# SAVE_DEPTH = os.path.join(SAVE_ROOT, 'depth')
SAVE_POSES = os.path.join(SAVE_ROOT, argv[4]+"_poses")



df = pd.read_csv(POSES)
col = df.columns
pose = []

def ks(x):
    return int(x[:-4])

files = sorted(os.listdir(COLOR), key=ks)
for i,f in enumerate(files):
    if i%FREQ!=0:
        continue
    img = Image.open(os.path.join(COLOR,f))
    img = img.resize((640, 480))
    img.save(os.path.join(SAVE_COLOR,f))



# files = sorted(os.listdir(DEPTH), key=ks)
# for i,f in enumerate(files):
#     if i%FREQ!=0:
#         continue
#     img = Image.open(os.path.join(DEPTH,f))
#     img = img.resize((640, 480))
#     img.save(os.path.join(SAVE_DEPTH,f))


for i,f in enumerate(files):
    if i%FREQ!=0:
        continue
    pose.append([df[col[1]][i], df[col[3]][i]])

pose = np.array(pose)
np.save(SAVE_POSES,pose)