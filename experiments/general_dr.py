import numpy as np
from k_medoids import kMedoids
from sklearn.svm import SVC

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

	# create cluster labels for each point
#	labels = np.zeros((N,))
#	for key in C.keys():
#		for ind in C[key]:
#			labels[ind] = (-1)**key
	labels_dict = {}
	for key in C.keys():
		str_key = str(key)
		for ind in C[key]:
			labels_dict[ind] = str_key
	labels = [labels_dict[j] for j in range(N)]

	# create maximum-margin separating hyperplane between original two clusters
	svm = SVC(kernel="linear")
	svm.fit(eigVecs, labels)
	print(svm.fit_status_)
	print(svm.predict(eigVecs))
	for key in C.keys():
		for ind in C[key]:
			assert svm.decision_function(np.reshape(eigVecs[ind,:], (1, 3)))[0] == str(key)
	print(dir(svm))
	print(svm.support_)
	print(svm.support_vectors_)

	# reflect all vectors in second cluster
	newEigVecs = eigVecs.copy()
	for ind in C[1]:
		newEigVecs[ind, :] = -eigVecs[ind, :]

	return np.mean(newEigVecs, axis=0)
