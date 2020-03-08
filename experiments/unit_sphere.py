import numpy as np

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

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def draw_spherical_mesh(ax, n_phi_rots=20, n_theta_rots=40):
	phi = np.linspace(0, np.pi, n_phi_rots)
	theta = np.linspace(0, 2 * np.pi, n_theta_rots)
	x = np.outer(np.sin(theta), np.cos(phi))
	y = np.outer(np.sin(theta), np.sin(phi))
	z = np.outer(np.cos(theta), np.ones_like(phi))
	ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)

def draw_spherical_data(sphere_jit, wire=True):
	xi, yi, zi = sphere_jit

	fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
	if wire:
		draw_spherical_mesh(ax)
	ax.scatter(xi, yi, zi, s=100, c='r', zorder=10)
	plt.show()

# The below is taken from DSM's answer on the following
# Stack Overflow post:
# https://stackoverflow.com/questions/4622057/plotting-3d-polygons-in-python-matplotlib

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import product

def draw_neighborhoods_from_pca(Ms, eigVecs):
	fig = plt.figure()
	ax = Axes3D(fig)
	for key in Ms.keys():
		m = Ms[key]
		u, v = eigVecs[key]
		uu = 0.5*u
		vv = 0.5*v
		ps = []
		ps.append(m + uu + vv)
		ps.append(m + uu - vv)
		ps.append(m - uu - vv)
		ps.append(m - uu + vv)
		ax.add_collection3d(Poly3DCollection([ps]))
	return ax
