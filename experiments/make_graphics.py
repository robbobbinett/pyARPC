import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl
import manifold_sampling as ms

def psdfr_eigvals_graphic():
	d = 50
	k = 10

	points = ms.sample_psd_fixed_rank(50, 10, 100000)
	p = points[0, :, :]

	try:
	        with open("data/psdfr_distances.pkl", "rb") as f:
	                dist_from_p = pkl.load(f)
	except FileNotFoundError:
	        dist_from_p = []
	        print("Populating distance vector...")
	        for j in tqdm(range(points.shape[0])):
	                dist_from_p.append(np.linalg.norm(p-points[j, :, :]))
	        dist_from_p = np.array(dist_from_p)
	        with open("data/psdfr_distances.pkl", "wb") as f:
	                pkl.dump(dist_from_p, f)

	radii = np.arange(np.sort(dist_from_p)[10], 40, 0.05)
	eigVals = []
	eigVecs = []

	for rad in tqdm(radii):
	        ps = points[dist_from_p < rad, :, :]
	        pps = np.reshape(ps, (ps.shape[0], 500))
	        cov = np.cov(pps, rowvar=False)
	        vals, vecs = np.linalg.eigh(cov)
	        eigVals.append(vals)
	        eigVecs.append(vecs)

	eigVals = np.stack(eigVals)
	eigVecs = np.stack(eigVecs)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	for j in range(d):
	        ax.plot(radii, eigVals[:, j])

	fig.savefig("graphics/psdfr_eigvals.pdf")

def rotation_eigen_plots():
	d = 10
	n = 10000

	points = ms.sample_rotations_manifold(d, n)
	p = points[0, :, :]

	try:
	        with open("data/rotations_distances.pkl", "rb") as f:
	                dist_from_p = pkl.load(f)
	except FileNotFoundError:
	        dist_from_p = []
	        print("Populating distance vector...")
	        for j in tqdm(range(points.shape[0])):
	                dist_from_p.append(np.linalg.norm(p-points[j, :, :]))
	        dist_from_p = np.array(dist_from_p)
	        with open("data/rotations_distances.pkl", "wb") as f:
	                pkl.dump(dist_from_p, f)

	radii = np.arange(np.sort(dist_from_p)[10], 6, 0.01)
	eigVals = []
	eigVecs = []

	print(np.sort(dist_from_p))

	for rad in tqdm(radii):
	        ps = points[dist_from_p < rad, :, :]
	        pps = np.reshape(ps, (ps.shape[0], d**2))
	        cov = np.cov(pps, rowvar=False)
	        vals, vecs = np.linalg.eigh(cov)
	        eigVals.append(vals)
	        eigVecs.append(vecs)

	eigVals = np.stack(eigVals)
	eigVecs = np.stack(eigVecs)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	for j in range(d):
	        ax.plot(radii, eigVals[:, j])

	fig.savefig("graphics/rotations_eigvals.pdf")
