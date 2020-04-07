import numpy as np
from k_medoids import kMedoids
from general_ortho import ortho_basis_of_span

def reduce_eigendicts(eigVals, eigVecs, d):
	new_vals = {}
	new_vecs = {}
	for key in eigVals.keys():
		vals = [(ind, val) for ind, val in enumerate(eigVals[key])]
		vals.sort(key=lambda x: x[1], reverse=True)
		new_vals[key] = [val[1] for val in vals[0:d]]
		vecs = [(ind, val) for ind, val in enumerate(eigVecs[key])]
		new_vecs[key] = [vecs[val[0]][1] for val in vals[0:d]]

	return new_vals, new_vecs

def choose_mean_quotient_eigenvector(eigVecs):
	"""
	This generates the mean eigenvector from a distribution of
	normalized eigenvectors. The routine used to generate these
	normalized eigenvectors---np.linalg.eigh---does not account
	for that there exist exactly two unit-length eigenvectors
	corresponding to each eigenspace.

	This routine finds the mean eigenvector after reflecting one
	of the two clusters of eigenvectors such that the dot product
	of any two eigenvectors in the sample post-translation is
	positive.

	Clustering is done by 2-medoids.
	"""
	# get number of eigenvectors
	N = eigVecs.shape[0]

	# populate distance matrix
	dist_mat = np.zeros((N, N))
	for j in range(N):
		for k in range(N):
			dist_mat[j, k] = np.linalg.norm(eigVecs[j,:]-eigVecs[k,:])

	# get clusters from 2-medoids
	_, C = kMedoids(dist_mat, 2)

	# reflect all vectors in second cluster
	newEigVecs = eigVecs.copy()
	for ind in C[1]:
		newEigVecs[ind, :] = -eigVecs[ind, :]

	meanQuotVec = np.mean(newEigVecs, axis=0)
	return meanQuotVec/np.linalg.norm(meanQuotVec)
