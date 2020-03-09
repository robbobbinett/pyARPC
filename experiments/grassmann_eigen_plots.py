import numpy as np
from manifold_sampling import sample_grassmann
import pickle as pkl
from tqdm import tqdm

d = 11

points = sample_grassmann(11, 5, 10000)
p = points[0, :, :]

try:
	with open("data/grassmann_distances.pkl", "rb") as f:
		dist_from_p = pkl.load(f)
except FileNotFoundError:
	dist_from_p = []
	print("Populating distance vector...")
	for j in tqdm(range(points.shape[0])):
		dist_from_p.append(np.linalg.norm(p-points[j, :, :]))
	dist_from_p = np.array(dist_from_p)
	with open("data/grassmann_distances.pkl", "wb") as f:
		pkl.dump(dist_from_p, f)

radii = np.arange(np.sort(dist_from_p)[10], 4, 0.01)
eigVals = []
eigVecs = []

print(np.sort(dist_from_p))

for rad in tqdm(radii):
	ps = points[dist_from_p < rad, :, :]
	pps = np.reshape(ps, (ps.shape[0], 55))
	cov = np.cov(pps, rowvar=False)
	vals, vecs = np.linalg.eigh(cov)
	eigVals.append(vals)
	eigVecs.append(vecs)

eigVals = np.stack(eigVals)
eigVecs = np.stack(eigVecs)

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
for j in range(d):
	ax.plot(radii, eigVals[:, j])

fig.savefig("graphics/grassmann_eigvals.pdf")
