from sys import argv
import re
import os
from PIL import Image

netvlad_pred_path = argv[1]
img_pair_save_path = argv[2]
choose_pred_count = 5
q_path = argv[3]
r_path = argv[4]

query_wise = os.path.join(img_pair_save_path,"query-wise")
retrieved_idx_wise = os.path.join(img_pair_save_path, "retrieved_idx_wise")
netvlad_best_wise = os.path.join(img_pair_save_path, "netvlad_best_wise")

if not os.path.exists(query_wise):
    os.mkdir(query_wise)
if not os.path.exists(retrieved_idx_wise):
    os.mkdir(retrieved_idx_wise)
if not os.path.exists(netvlad_best_wise):
    os.mkdir(netvlad_best_wise)

pred_f = open(netvlad_pred_path,'r')
pred = pred_f.readlines()
pred_f.close()

qr = {}
for i, line in enumerate(pred):
    [q, r] = [int(s) for s in re.findall(r'-?\d+', line)]
    if q in qr.keys():
        if len(qr[q])<int(choose_pred_count):
            qr[q].append(r)
    else:
        qr[q]=[r]

for k in qr.keys():
    for r in qr[k]:
        if not os.path.exists(os.path.join(query_wise, str(k))):
            os.mkdir(os.path.join(query_wise, str(k)))

        fname = str(k)+"-"+str(r)+".png"
        fname = os.path.join(query_wise, str(k), fname)

        #Read the two images
        image1 = Image.open(os.path.join(q_path,str(k)+".jpg"))
        image2 = Image.open(os.path.join(r_path,str(r)+".jpg"))
        # image1.show()
        # image2.show()
        #resize, first image
        # image1 = image1.resize((426, 240))
        image1_size = image1.size
        image2_size = image2.size
        new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
        new_image.paste(image1,(0,0))
        new_image.paste(image2,(image1_size[0],0))
        new_image.save(fname,"JPEG")
        # new_image.show()
        # break
    # break




for k in qr.keys():
    for r in qr[k]:
        if not os.path.exists(os.path.join(retrieved_idx_wise, str(r))):
            os.mkdir(os.path.join(retrieved_idx_wise, str(r)))

        fname = str(k)+"-"+str(r)+".png"
        fname = os.path.join(retrieved_idx_wise, str(r), fname)

        #Read the two images
        image1 = Image.open(os.path.join(q_path,str(k)+".jpg"))
        image2 = Image.open(os.path.join(r_path,str(r)+".jpg"))
        # image1.show()
        # image2.show()
        #resize, first image
        # image1 = image1.resize((426, 240))
        image1_size = image1.size
        image2_size = image2.size
        new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
        new_image.paste(image1,(0,0))
        new_image.paste(image2,(image1_size[0],0))
        new_image.save(fname,"JPEG")
        # new_image.show()
        # break
    # break


for l in range(choose_pred_count):
    for k in qr.keys():
        r = qr[k][l]
        if not os.path.exists(os.path.join(netvlad_best_wise, str(l))):
            os.mkdir(os.path.join(netvlad_best_wise, str(l)))

        fname = str(k)+"-"+str(r)+".png"
        fname = os.path.join(netvlad_best_wise, str(l), fname)

        #Read the two images
        image1 = Image.open(os.path.join(q_path,str(k)+".jpg"))
        image2 = Image.open(os.path.join(r_path,str(r)+".jpg"))
        # image1.show()
        # image2.show()
        #resize, first image
        # image1 = image1.resize((426, 240))
        image1_size = image1.size
        image2_size = image2.size
        new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
        new_image.paste(image1,(0,0))
        new_image.paste(image2,(image1_size[0],0))
        new_image.save(fname,"JPEG")
        # new_image.show()
        # break
    # break