import numpy as np
from time import time
from tqdm import tqdm

n = 2000
d = 3

import pickle as pkl
try:
	with open("data/unit_sphere_with_jitter.pkl", "rb") as f:
		sphere_jit = pkl.load(f)
except FileNotFoundError:
	from unit_sphere import sample_spherical
	sphere = sample_spherical(n, d)
	sphere_jit = sphere + 0.1*np.random.randn(d, n)
	with open("data/unit_sphere_with_jitter.pkl", "wb") as f:
		pkl.dump(sphere_jit, f)

from unit_sphere import draw_spherical_data
#draw_spherical_data(sphere_jit)

from k_medoids import kMedoids
try:
	with open("data/unit_sphere_dist_mat.pkl", "rb") as f:
		dist_mat = pkl.load(f)
except FileNotFoundError:
	print("Populating distance matrix...")
	dist_mat = np.zeros((n, n))
	for j in range(n):
		for k in range(n):
			dist_mat[j, k] = np.linalg.norm(sphere_jit[:, j]-sphere_jit[:, k])
	with open("data/unit_sphere_dist_mat.pkl", "wb") as f:
		pkl.dump(dist_mat, f)

p = sphere_jit[:, 0]
dist_from_p = dist_mat[0, :]
radii = np.arange(0.2, 2.0, 0.01)

eigVals = []
eigVecs = []

for rad in tqdm(radii):
	ps = sphere_jit[:, dist_from_p < rad]
	cov = np.cov(ps)
	vals, vecs = np.linalg.eigh(cov)
	eigVals.append(vals)
	eigVecs.append(vecs)

eigVals = np.stack(eigVals)
eigVecs = np.stack(eigVecs)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gif

fig = plt.figure()
ax = fig.add_subplot(111)
for j in range(d):
	ax.plot(radii, eigVals[:, j])

fig.savefig("graphics/eigvals.pdf")

eigVecsX = eigVecs[:, 0, :]
eigVecsY = eigVecs[:, 1, :]
eigVecsZ = eigVecs[:, 2, :]

@gif.frame
def plot_vec(ind, k):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	ax.plot(eigVecsX[0:ind, k], eigVecsY[0:ind, k], eigVecsZ[0:ind, k], "bs")
	ax.set_xlim(-1, 1)
	ax.set_ylim(-1, 1)
	ax.set_zlim(-1, 1)

for j in range(3):
	frames = []
	for ind in range(eigVecs.shape[0]):
		frame = plot_vec(ind, j)
		frames.append(frame)
	gif.save(frames, "graphics/eigenvecs_"+str(j)+".gif", duration=200)
