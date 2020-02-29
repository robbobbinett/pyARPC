import numpy as np
from time import time

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

def k_fig(k, show=False):
	M, C = kMedoids(dist_mat, k)
	if show:
		phi = np.linspace(0, np.pi, 20)
		theta = np.linspace(0, 2 * np.pi, 40)
		x = np.outer(np.sin(theta), np.cos(phi))
		y = np.outer(np.sin(theta), np.sin(phi))
		z = np.outer(np.cos(theta), np.ones_like(phi))

		fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
		for key in C.keys():
			xi, yi, zi = sphere_jit[:, np.array(C[key])]

			ax.scatter(xi, yi, zi, s=100, zorder=10)

		plt.show()
	return M, C

M, C = k_fig(10)

Ms = dict((j, sphere_jit[:, m]) for j, m in enumerate(M))
Cs = dict((j, sphere_jit[:,C[j]]) for j in range(len(M)))

Covs = dict((j, np.cov(Cs[j])) for j in range(len(M)))

eigVals = {}
eigVecs = {}
for j in range(len(M)):
	eigVals[j], eigVecs[j] = np.linalg.eigh(Covs[j])

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from unit_sphere import draw_spherical_mesh, draw_neighborhoods_from_pca

draw_neighborhoods_from_pca(Ms, eigVals, eigVecs)

fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
draw_spherical_mesh(ax, n_phi_rots=10, n_theta_rots=20)
plt.show()
