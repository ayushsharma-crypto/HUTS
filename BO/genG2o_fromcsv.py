'''
Transformation convention: 
T2_1 : 2 wrt 1

Run cmd : 

python genG2o_fromcsv.py ../../data/SMALL/Reference_Poses.csv ../../data/SMALL/Query_Poses.csv ../../data/SMALL/RoRD_new/rord_matches_count.txt  ../../data/SMALL/RoRD_new/transition/
python genG2o_fromcsv.py ../../data/subsequences/combined/R_Poses.csv ../../data/subsequences/combined/Q_Poses.csv ../../data/subsequences/combined/rord_matches_count.txt  ../../data/subsequences/combined/transition/    > tmp.txt
'''


import matplotlib.pyplot as plt
from sys import argv, exit
import math
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from mpl_toolkits import mplot3d
import pandas as pd
import re


def readPose(db_filename, q_filename):
	db_poses = pd.read_csv(db_filename)
	db_col = db_poses.columns

	len_DB = np.array(range(len(db_poses[db_col[1]])))
	X = np.array(db_poses[db_col[1]][len_DB])
	Y = np.array(db_poses[db_col[2]][len_DB])
	Z = np.array(db_poses[db_col[3]][len_DB])
	Qw = np.array(db_poses[db_col[4]][len_DB])
	Qx = np.array(db_poses[db_col[5]][len_DB])
	Qy = np.array(db_poses[db_col[6]][len_DB])
	Qz = np.array(db_poses[db_col[7]][len_DB])
	

	q_poses = pd.read_csv(q_filename)
	q_col = q_poses.columns

	len_Q = np.array(range(len(q_poses[q_col[1]])))
	X = np.concatenate((X,np.array(q_poses[q_col[1]][len_Q])))
	Y = np.concatenate((Y,np.array(q_poses[q_col[2]][len_Q])))
	Z = np.concatenate((Z,np.array(q_poses[q_col[3]][len_Q])))
	Qw = np.concatenate((Qw,np.array(q_poses[q_col[4]][len_Q])))
	Qx = np.concatenate((Qx,np.array(q_poses[q_col[5]][len_Q])))
	Qy = np.concatenate((Qy,np.array(q_poses[q_col[6]][len_Q])))
	Qz = np.concatenate((Qz,np.array(q_poses[q_col[7]][len_Q])))


	rot_angle = 2*np.arcsin(np.around(2*Qw*Qy,decimals=2))
	return X, Z, rot_angle, len(len_DB), len(len_Q)



def draw(X, Y, THETA):
	ax = plt.subplot(111)
	ax.plot(X, Y, 'go')
	ax.plot(X, Y, 'k-')
	ax.set_aspect('auto', 'datalim')
	ax.margins(0.1)
	ax.set_xlabel("X")
	ax.set_ylabel("Z")

	plt.show()


def drawTheta(X, Y, THETA):
	ax = plt.subplot(111)
	for i in range(len(THETA)):
		x2 = math.cos(THETA[i]) + X[i]
		y2 = math.sin(THETA[i]) + Y[i]
		plt.plot([X[i], x2], [Y[i], y2], 'm->')

	ax.plot(X, Y, 'ro')	
	ax.set_xlabel("X")
	ax.set_ylabel("Z")
	plt.show()

def drawTwo(X, Y, X2, Y2):
	ax = plt.subplot(111)
	ax.plot(X, Y, 'go')
	ax.plot(X, Y, 'k-')
	ax.plot(X2, Y2, 'ro')
	ax.plot(X2, Y2, 'k-')
	ax.set_aspect('auto', 'datalim')
	ax.margins(0.1)
	ax.set_xlabel("X")
	ax.set_ylabel("Z")

	plt.show()



def addNoise(X, Y, THETA):
	xN = np.zeros(len(X)); yN = np.zeros(len(Y)); tN = np.zeros(len(THETA))
	xN[0] = X[0]; yN[0] = Y[0]; tN[0] = THETA[0]

	for i in range(1, len(X)):
		# Get T2_1
		p1 = (X[i-1], Y[i-1], THETA[i-1])
		p2 = (X[i], Y[i], THETA[i])
		T1_w = np.array([[math.cos(p1[2]), -math.sin(p1[2]), p1[0]], [math.sin(p1[2]), math.cos(p1[2]), p1[1]], [0, 0, 1]])
		T2_w = np.array([[math.cos(p2[2]), -math.sin(p2[2]), p2[0]], [math.sin(p2[2]), math.cos(p2[2]), p2[1]], [0, 0, 1]])
		T2_1 = np.dot(np.linalg.inv(T1_w), T2_w)
		del_x = T2_1[0][2]
		del_y = T2_1[1][2]
		del_theta = math.atan2(T2_1[1, 0], T2_1[0, 0])
		
		# Add noise
		if(i<5):
			xNoise = 0; yNoise = 0; tNoise = 0
		else:
			# np.random.seed(42)
			# xNoise = np.random.normal(0, 0.08); yNoise = np.random.normal(0, 0.08); tNoise = np.random.normal(0, 0.002)
			# xNoise = 0.005; yNoise = 0.005; tNoise = -0.0005
			# xNoise = 0.01; yNoise = 0.01; tNoise = 0.0007
			xNoise = 0.01; yNoise = 0.01; tNoise = 0.00015


		del_xN = del_x + xNoise; del_yN = del_y + yNoise; del_thetaN = del_theta + tNoise

		# Convert to T2_1'
		T2_1N = np.array([[math.cos(del_thetaN), -math.sin(del_thetaN), del_xN], [math.sin(del_thetaN), math.cos(del_thetaN), del_yN], [0, 0, 1]])

		# Get T2_w' = T1_w' . T2_1'
		p1 = (xN[i-1], yN[i-1], tN[i-1])
		T1_wN = np.array([[math.cos(p1[2]), -math.sin(p1[2]), p1[0]], [math.sin(p1[2]), math.cos(p1[2]), p1[1]], [0, 0, 1]])
		T2_wN = np.dot(T1_wN, T2_1N)
		
		# Get x2', y2', theta2'
		x2N = T2_wN[0][2]
		y2N = T2_wN[1][2]
		theta2N = math.atan2(T2_wN[1, 0], T2_wN[0, 0])

		xN[i] = x2N; yN[i] = y2N; tN[i] = theta2N

	# tN = getTheta(xN, yN)

	return (xN, yN, tN)



def writeG2O(X, Y, THETA, file):
	g2o = open(file, 'w')

	for i, (x, y, theta) in enumerate(zip(X, Y, THETA)):
		line = "VERTEX_SE2 " + str(i) + " " + str(x) + " " + str(y) + " " + str(theta)
		g2o.write(line)
		g2o.write("\n")

	# Odometry
	# T1_w : 1 with respect to world
	g2o.write("# Odometry constraints\n")
	info_mat = "500.0 0.0 0.0 500.0 0.0 500.0"
	for i in range(1, len(X)):
		p1 = (X[i-1], Y[i-1], THETA[i-1])
		p2 = (X[i], Y[i], THETA[i])
		T1_w = np.array([[math.cos(p1[2]), -math.sin(p1[2]), p1[0]], [math.sin(p1[2]), math.cos(p1[2]), p1[1]], [0, 0, 1]])
		T2_w = np.array([[math.cos(p2[2]), -math.sin(p2[2]), p2[0]], [math.sin(p2[2]), math.cos(p2[2]), p2[1]], [0, 0, 1]])
		T2_1 = np.dot(np.linalg.inv(T1_w), T2_w)
		del_x = str(T2_1[0][2])
		del_y = str(T2_1[1][2])
		del_theta = str(math.atan2(T2_1[1, 0], T2_1[0, 0]))

		line = "EDGE_SE2 "+str(i-1)+" "+str(i)+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat+"\n"
		g2o.write(line)

	g2o.write("FIX 0\n")
	g2o.close()



def right2left():
	"""
	Returns: Right wrt left = TL_R = TR2L

	LHS -> Ry(-90) -> Rx(90) -> RHS

	TL_R = Ry(-90) Rx(90)
	
	"""


	thetaX = math.radians(90)
	thetaY = math.radians(-90)

	Rx = np.array([[1, 0, 0], [0, math.cos(thetaX), -math.sin(thetaX)], [0, math.sin(thetaX), math.cos(thetaX)]])
	Ry = np.array([[math.cos(thetaY), 0, math.sin(thetaY)], [0, 1, 0], [-math.sin(thetaY), 0, math.cos(thetaY)]])

	TR2L = np.identity(4)
	TR2L[0:3, 0:3] =  Ry @ Rx 
	
	return TR2L



def L2R(TC1C2L):
	TLR = right2left()
	TC1C2R = np.linalg.inv(TLR) @ TC1C2L @ TLR
	# Camera wrt Base in Right hand frame
	TBC = np.eye(4)
	TBC[1,3]+=1
	# B2 wrt B1 in Right hand frame
	TB1B2R = TBC @ TC1C2R @ np.linalg.inv(TBC)
	return TB1B2R



if __name__ == '__main__':
	dirc = os.path.dirname(argv[1])
	X, Z, rot_angle, len_db, len_q = readPose(argv[1], argv[2])
	# draw(X, Z, rot_angle)
	writeG2O(X, Z, rot_angle, os.path.join(os.path.join(dirc,"PGO"), "gt.g2o"))

	(XN, ZN, rot_angleN) = addNoise(X, Z, rot_angle)
	writeG2O(XN, ZN, rot_angleN, os.path.join(os.path.join(dirc,"PGO"), "noise.g2o"))

	# drawTwo(X, Z, XN, ZN)
	# drawTheta(X,Z,rot_angle)
	# drawTheta(XN,ZN,rot_angleN)

	assert len(X)==len(XN)
	assert len(Z)==len(ZN)
	assert len(rot_angle)==len(rot_angleN)
	
	# writing loop_pair.txt

	RM = open(argv[3], "r")
	RML = RM.readlines()
	RM.close()
	best_dict = {}
	for l in RML:
		l = l.rstrip()
		(_, rgb1, _, rgb2, count) = l.split()
		count = int(count)
		idx1 = (int(re.search('\d+', rgb1)[0]) - 1) + len_db
		idx2 = int(re.search('\d+', rgb2)[0]) - 1
		# uncomment for small_2
		idx1 = (int(re.findall('\d+', rgb1)[2]) - 1) + len_db
		idx2 = int(re.findall('\d+', rgb2)[2]) - 1
		if idx1 in best_dict.keys():
			[rid, ct] = best_dict[idx1]
			if ct<count:
				best_dict[idx1]=[idx2, count]
		else:
			best_dict[idx1]=[idx2, count]
	
	for keys in best_dict.keys():
		print(keys, " = ", best_dict[keys])

	# Write filter....for more accurate trnsition edges!
	# loop_pair_error = open(argv[5],'r')
	# loop_pair_error_lines = loop_pair_error.readlines()
	# loop_pair_error.close()
	# for line in loop_pair_error_lines:
	# 	line = line.strip()
	# 	[idx, q, r, rot_err, t_err] = line.split()
	# 	print(line.split())
	# 	if float(rot_err)>60 or float(t_err)>3:
	# 		print("Higher error.... ", int(q)+len_db)
	# 		if int(q)+len_db in best_dict.keys():
	# 			print("Query exist in best dict....")
	# 			if best_dict[int(q)+len_db][0]==int(r):
	# 				print("Definite query removed")
	# 				best_dict.pop(int(q)+len_db)
		


	transition_file = argv[4]
	loop_pair_file = open(os.path.join(os.path.join(dirc,"PGO"), "loop_pairs.txt"), 'w')
	
	print("\n\nUsing these loop pairs only:- ", len(best_dict.keys()))
	for keys in best_dict.keys():
		print(keys, " ", best_dict[keys])
	print("\n\nUsing these loop pairs only (in query & db idx):-")
	for keys in best_dict.keys():
		print(keys-len_db+1, " [", best_dict[keys][0]+1,  best_dict[keys][1],"]")

	print("\n")

	for idx1 in best_dict.keys():
		[idx2,_] = best_dict[idx1]
		# db idx          query idx    - for g2o file
		T_pred = np.load(os.path.join(transition_file,str(idx1+1-len_db)+"-"+str(1+idx2)+".npy"))
		T_pred = L2R(T_pred)
		[dx, dy, dz] = T_pred[:3,3]
		[dqx, dqy, dqz, dqw] = R.from_matrix(T_pred[:3,:3]).as_quat()
		loop_pair_file.write(str(idx1)+" "+ str(idx2)+"\n")
		loop_pair_file.write(str(dx)+" ")
		# loop_pair_file.write(str(dy)+" ")
		loop_pair_file.write(str(dz)+" ")
		# loop_pair_file.write(str(dqx)+" ")
		# loop_pair_file.write(str(dqy)+" ")
		# loop_pair_file.write(str(dqz)+" ")
		# loop_pair_file.write(str(dqw)+"\n")
		loop_pair_file.write(str(2*np.arcsin(2*dqw*dqy))+"\n")
	loop_pair_file.close()
