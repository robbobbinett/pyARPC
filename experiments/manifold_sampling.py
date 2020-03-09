import numpy as np
from pymanopt.manifolds import Grassmann, PSDFixedRank
import pickle as pkl
from tqdm import tqdm

def sample_grassmann(d, p, n):
	"""
	Sample n points from the manifold of p-dimensional subspaces
	of R^d using pymanopt.manifolds.grassmann.Grassmann.rand
	"""
	try:
		with open("data/grassmann__"+str(d)+"_"+str(p)+"_"+str(n)+".pkl", "rb") as f:
			points = pkl.load(f)
	except FileNotFoundError:
		print("Sampling "+str(n)+" points from Grassmann(+"+str(d)+","+str(p)+")...")
		manifold = Grassmann(d, p)
		points = []
		for _ in tqdm(range(n)):
			points.append(manifold.rand())
		points = np.stack(points)
		with open("data/grassmann__"+str(d)+"_"+str(p)+"_"+str(n)+".pkl", "wb") as f:
			pkl.dump(points, f)
	return points

def sample_psd_fixed_rank(d, k, n):
	"""
	Sample n points from the manifold of n x n PSD matrices of
	rank k using pymanopt.manifolds.psd.PSDFixedRank.rand
	"""
	try:
		with open("data/psdfr__"+str(d)+"_"+str(k)+"_"+str(n)+".pkl", "rb") as f:
			points = pkl.load(f)
	except FileNotFoundError:
		print("Sampling "+str(n)+" points from PSDFixedRank("+str(n)+","+str(k)+")...")
		manifold = PSDFixedRank(d, k)
		points = []
		for _ in tqdm(range(n)):
			points.append(manifold.rand())
		points = np.stack(points)
		with open("data/psdfr__"+str(d)+"_"+str(k)+"_"+str(n)+".pkl", "wb") as f:
			pkl.dump(points, f)
	return points

sample_grassmann(11, 5, 1000)
sample_psd_fixed_rank(50, 10, 1000)
