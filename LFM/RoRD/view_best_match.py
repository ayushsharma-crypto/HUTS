import os
import numpy as np
from sys import argv
from PIL import  Image
import re

RORD_MATCHES_COUNT = argv[1]
FETCH_PATH = argv[2]
SAVE_BEST_PATH = argv[3]
QUERY_WISE_SAVE_PATH = argv[4]


if not os.path.exists(QUERY_WISE_SAVE_PATH):
    os.mkdir(QUERY_WISE_SAVE_PATH)



if not os.path.exists(SAVE_BEST_PATH):
    os.mkdir(SAVE_BEST_PATH)



RM = open(RORD_MATCHES_COUNT, "r")
RML = RM.readlines()
RM.close()

best_dict = {}

for l in RML:
    l = l.rstrip()
    (_, rgb1, _, rgb2, count) = l.split()
    count = int(count)
    idx1 = int(re.findall('\d+', rgb1)[2]) 
    idx2 = int(re.findall('\d+', rgb2)[2]) 
    # print(idx1, idx2)
    if idx1 in best_dict.keys():
        [rid, ct] = best_dict[idx1]
        if ct<count:
            best_dict[idx1]=[idx2, count]
    else:
        best_dict[idx1]=[idx2, count]
    

# for K in best_dict.keys():
#     print(K," ", best_dict[K])


for K in best_dict.keys():
    [rid, ct] = best_dict[K]
    f = os.path.join(FETCH_PATH,str(K)+"-"+str(rid)+".jpg")
    img = Image.open(f)
    f = os.path.join(SAVE_BEST_PATH,str(K)+"-"+str(rid)+".jpg")
    img.save(f)

best_dict = {}

for l in RML:
    l = l.rstrip()
    (_, rgb1, _, rgb2, count) = l.split()
    count = int(count)
    idx1 = int(re.findall('\d+', rgb1)[2]) 
    idx2 = int(re.findall('\d+', rgb2)[2]) 
    f = os.path.join(FETCH_PATH,str(idx1)+"-"+str(idx2)+".jpg")
    img = Image.open(f)
    FOLDER = os.path.join(QUERY_WISE_SAVE_PATH, str(idx1))
    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)   
    f = os.path.join(FOLDER,str(idx1)+"-"+str(idx2)+".jpg") 
    img.save(f)    