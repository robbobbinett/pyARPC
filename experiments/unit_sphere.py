fig_0 = False
wire = False

import numpy as np
from time import time

# The below is taken from ali_m's answer on the following
# Stack Overflow post:
# https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere
def sample_spherical(npoints, ndim=3):
	"""
	Randomly sample points uniformly from the unit ball
	"""
	vec = np.random.randn(ndim, npoints)
	vec /= np.linalg.norm(vec, axis=0)
	return vec

n = 100
d = 3

sphere = sample_spherical(n, d)
sphere_jit = sphere + 0.1*np.random.randn(d, n)

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

if fig_0:
	phi = np.linspace(0, np.pi, 20)
	theta = np.linspace(0, 2 * np.pi, 40)
	x = np.outer(np.sin(theta), np.cos(phi))
	y = np.outer(np.sin(theta), np.sin(phi))
	z = np.outer(np.cos(theta), np.ones_like(phi))

	xi, yi, zi = sphere_jit

	#fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
	fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
	if wire:
		ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
	ax.scatter(xi, yi, zi, s=100, c='r', zorder=10)

	plt.show()

from k_medoids import kMedoids
start = time()
dist_mat = np.zeros((n, n))
for j in range(n):
	for k in range(n):
		dist_mat[j, k] = np.linalg.norm(sphere_jit[:, j]-sphere_jit[:, k])

def k_fig(k):
	M, C = kMedoids(dist_mat, k)
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
f = plt.figure()
ax = f.add_subplot(111)
for m in M:
	ax.plot(sphere_jit[:, m])
plt.show()
