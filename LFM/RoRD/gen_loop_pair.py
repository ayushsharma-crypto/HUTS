from sys import argv
import re



netvlad_pred_path = argv[1]
loop_pair_path = argv[2]
register_cmd_path = argv[3]
choose_pred_count = argv[4]

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

lp = open(loop_pair_path, 'w')
cm = open(register_cmd_path, 'w')
cm.write("#!/bin/sh\n\n")
for i,k in enumerate(qr.keys()):
    if i%2:
        continue
    for idx,r in enumerate(qr[k]):
        lp.write(str(k)+' '+str(r)+'\n')

        cm.write("\necho "+str(k)+' '+str(r)+'\n')

        cmd = "python register_original.py "
        cmd = cmd + " --rgb1  ../../../original_data/small_2/query1/color/" + str(k) + ".jpg"
        cmd = cmd + " --depth1  ../../../original_data/small_2/query1/depth/" + str(k) + ".png"
        cmd = cmd + " --rgb2  ../../../original_data/small_2/reference1/color/" + str(r) + ".jpg"
        cmd = cmd + " --depth2  ../../../original_data/small_2/reference1/depth/" + str(r) + ".png"
        cmd = cmd + " --camera_file config/camera_habitat_original.txt "
        cmd = cmd + " --model_rord ../../../data/small_2/RoRD/rord.pth "
        cmd = cmd + " --H config/topH_original.npy "
        cmd = cmd + " --save_trans ../../../data/small_2/RoRD/transition/"+str(k)+"-"+str(r)+".npy" 
        cmd = cmd + " --save_persp ../../../data/small_2/RoRD/perspective/"+str(k)+"-"+str(r)+".jpg" 
        cmd = cmd + " --save_ortho ../../../data/small_2/RoRD/orthographic/"+str(k)+"-"+str(r)+".jpg" 
        cmd = cmd + " --save_matches ../../../data/small_2/RoRD/rord_matches_count.txt "
        cmd = cmd + " --save_desp ../../../data/small_2/RoRD/desp/" + str(k)+"-"+str(r)+".npy" 

        cm.write(cmd+"\n\n")

lp.close()
cm.close()