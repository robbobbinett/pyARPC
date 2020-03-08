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
	for j in tqdm(range(n)):
		for k in range(n):
			dist_mat[j, k] = np.linalg.norm(sphere_jit[:, j]-sphere_jit[:, k])
	with open("data/unit_sphere_dist_mat.pkl", "wb") as f:
		pkl.dump(dist_mat, f)

p = sphere_jit[:, 0]
dist_from_p = dist_mat[0, :]
radii = np.arange(0.01, 0.5, 0.01)

eigVals = []
eigVecs = []

print("Looking at eigenvals for varying radii...")
for rad in tqdm(radii):
	ps = sphere_jit[:, dist_from_p < rad]
	cov = np.cov(ps)
	vals, vecs = np.linalg.eigh(cov)
	eigVals.append(vals)
	eigVecs.append(vecs)

eigVals = np.stack(eigVals)
eigVecs = np.stack(eigVecs)

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
for j in range(d):
	ax.plot(radii, eigVals[:, j], color="b")

plt.show()
