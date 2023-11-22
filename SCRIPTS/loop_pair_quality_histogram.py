import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import matplotlib.pyplot as plt

def transformation_matrix_from_7(x1, y1, z1, qx1, qy1, qz1, qw1):
	r1 = R.from_quat([qx1, qy1, qz1, qw1])
	m1 = np.hstack((r1.as_matrix(), np.array([x1,y1,z1], dtype=np.float64).reshape(3,1)))
	m1 = np.vstack((m1, np.array([0,0,0,1], dtype=np.float64).reshape(1,4)))
	return m1

def quat2eulers(qw,qx,qy, qz):
	
	r = R.from_quat([qx, qy, qz, qw])
	roll, pitch, yaw = r.as_euler('xyz', degrees=True)
	return roll, pitch, yaw	

with open("qsmall_ours_lp1.txt", "r") as f:
	corr = f.readlines()
corr = [x.strip() for x in corr]
corr = sorted(corr, key=lambda x: float(x.split(" ")[2]))
corr = corr[:32]

with open("/home/commander/research/slam_frontend/Original Data/poses.csv", "r") as f:
	poses = f.readlines()
poses = [x.strip() for x in poses]

pitchlist = []
translationlist = []
skipped = 0
print("l1,l2,r,p,y,d")
for line in corr:
	l1 = int(line.split(" ")[0])
	l2 = int(line.split(" ")[1])
	_ , x1, y1, z1, qw1, qx1, qy1, qz1 = poses[3*l1 - 2].split(",")
	_ , x2, y2, z2, qw2, qx2, qy2, qz2 = poses[3*l2 - 2].split(",")
	r1, p1, yaw1 = quat2eulers(float(qw1), float(qx1), float(qy1), float(qz1))
	r2, p2, yaw2 = quat2eulers(float(qw2), float(qx2), float(qy2), float(qz2))
	x1 = float(x1)
	x2 = float(x2)
	y1 = float(y1)
	y2 = float(y2)
	z1 = float(z1)
	z2 = float(z2)
	# print(x1, x2, y1, y`2, z1, z2)
	# print(f"{l1},{l2},{r1-r2},{p1-p2},{y1-y2},{math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)}")
	delta_translation = math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
	if(delta_translation > 100):
		print(f"skipped {l1} {l2} because of large translation difference {x1} {x2} {y1} {y2} {z1} {z2}")
		skipped +=1
		continue
	pitchlist.append(abs(p1-p2))
	translationlist.append(math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2))

fig, ax = plt.subplots(2,2)

ax[0,0].hist(pitchlist, bins=3, density=True)
ax[0,0].set_title("angle difference, ours ")
# plt.show()
ax[0, 1].hist(translationlist, bins=25, density=True)
ax[0, 1].set_title("translation difference, ours")

with open("/home/commander/iiit/temp/rtabmap_qsmall_lp.txt", "r") as f:
	corr = f.readlines()
corr = [x.strip() for x in corr]

pitchlist = []
translationlist = []
skipped = 0
print("l1,l2,r,p,y,d")
for line in corr:
	l1 = int(line.split(" ")[0])
	l2 = int(line.split(" ")[1])
	_ , x1, y1, z1, qw1, qx1, qy1, qz1 = poses[l1].split(",")
	_ , x2, y2, z2, qw2, qx2, qy2, qz2 = poses[l2].split(",")
	r1, p1, yaw1 = quat2eulers(float(qw1), float(qx1), float(qy1), float(qz1))
	r2, p2, yaw2 = quat2eulers(float(qw2), float(qx2), float(qy2), float(qz2))
	x1 = float(x1)
	x2 = float(x2)
	y1 = float(y1)
	y2 = float(y2)
	z1 = float(z1)
	z2 = float(z2)
	# print(x1, x2, y1, y`2, z1, z2)
	# print(f"{l1},{l2},{r1-r2},{p1-p2},{y1-y2},{math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)}")
	delta_translation = math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
	if(delta_translation > 100):
		print(f"skipped {l1} {l2} because of large translation difference {x1} {x2} {y1} {y2} {z1} {z2}")
		skipped +=1
		continue
	pitchlist.append(abs(p1-p2))
	translationlist.append(math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2))
ax[1,0].hist(pitchlist, bins=3, density=True)
ax[1,0].set_title("angle difference, rtabmap")
# plt.show()
ax[1, 1].hist(translationlist, bins=25, density=True)
ax[1, 1].set_title("translation difference, rtabmap")
plt.show()