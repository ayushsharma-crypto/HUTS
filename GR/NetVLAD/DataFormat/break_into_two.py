import os
import numpy as np
import pandas as pd
from sys import argv
from PIL import  Image

BREAKPOINT = int(argv[1])
ROOT = argv[2]


COLOR = os.path.join(ROOT, 'color')
DEPTH = os.path.join(ROOT, 'depth')
POSES = os.path.join(ROOT, 'poses.csv')

QSAVE = os.path.join(ROOT, 'query')
QSAVECOLOR = os.path.join(QSAVE, 'color')
QSAVEDEPTH = os.path.join(QSAVE, 'depth')
RSAVE = os.path.join(ROOT, 'reference')
RSAVECOLOR = os.path.join(RSAVE, 'color')
RSAVEDEPTH = os.path.join(RSAVE, 'depth')

if not os.path.exists(QSAVE):
    os.mkdir(QSAVE)
if not os.path.exists(RSAVE):
    os.mkdir(RSAVE)

if not os.path.exists(QSAVECOLOR):
    os.mkdir(QSAVECOLOR)
if not os.path.exists(RSAVECOLOR):
    os.mkdir(RSAVECOLOR)

if not os.path.exists(QSAVEDEPTH):
    os.mkdir(QSAVEDEPTH)
if not os.path.exists(RSAVEDEPTH):
    os.mkdir(RSAVEDEPTH)

df = pd.read_csv(POSES)
col = df.columns

def ks(x):
    return int(x[:-4])

files = sorted(os.listdir(COLOR), key=ks)
for i,f in enumerate(files):
    img = Image.open(os.path.join(COLOR,f))
    if i<BREAKPOINT:
        img.save(os.path.join(RSAVECOLOR,f))
    else:
        f1 = str(int(f[:-4])-BREAKPOINT)+".jpg"
        img.save(os.path.join(QSAVECOLOR,f1))

files = sorted(os.listdir(DEPTH), key=ks)
for i,f in enumerate(files):
    img = Image.open(os.path.join(DEPTH,f))
    if i<BREAKPOINT:
        img.save(os.path.join(RSAVEDEPTH,f))
    else:
        f1 = str(int(f[:-4])-BREAKPOINT)+".png"
        img.save(os.path.join(QSAVEDEPTH,f1))


rpose = df[:BREAKPOINT]
qpose = df[BREAKPOINT:]
rpose.to_csv(os.path.join(RSAVE,'poses.csv'), index=False)
qpose.to_csv(os.path.join(QSAVE,'poses.csv'), index=False)