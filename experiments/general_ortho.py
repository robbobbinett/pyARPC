import numpy as np

def is_independent(list_of_vecs):
	"""
	Determines whether a list of vectors is linearly independent.
	Requires vectors to have same shape as (1D) NumPy arrays
	"""
	# make sure list_of_vecs is...well...a list
	if not isinstance(list_of_vecs, list):
		raise TypeError("list_of_vecs should be of type list; currently of type "+str(type(list_of_vecs))+".")
	# make sure list_of_vecs contains at least one element
	if len(list_of_vecs) == 0:
		raise ValueError("list_of_vecs should have nonzero length.")
	# make sure all vectors are of type np.ndarray
	if not all([isinstance(item, np.ndarray) for item in list_of_vecs]):
		raise TypeError("All elements of list_of_vecs should be of type np.ndarray; currently of types: "+", ".join([str(type(item)) for item in list_of_vecs]))
	# make sure first vector is an array of dimension 1
	if list_of_vecs[0].dim != 1:
		raise ValueError("The first element of list_of_vecs should be of dimension 1; currently of dimension "+str(list_of_vecs[0].dim)+".")
	# make sure all vectors have same shape
	init_shape = list_of_vecs[0].shape
	if not all([item.shape == init_shape for item in list_of_vecs]):
		raise ValueError("Vectors in list_of_vecs have varying shapes: "+", ".join([str(item.shape) for item in list_of_vecs]))

	# compute rank using np.linalg.matrix_rank
	mat_from_vecs = np.stack(list_of_vecs, axis=0)

	return len(list_of_vecs) == np.linalg.matrix_rank(mat_from_vecs)

def ortho_basis_of_span(list_of_vecs):
	"""
	Takes a list of linearly independent vectors (i.e. NumPy araays
	of dimension 1 with identical shape) and returns an orthonormal
	basis for their span. Generates span using np.linalg.qr
	"""
	# check for linear independence
	if not is_independent(list_of_vecs):
		raise ValueError("list_of_vecs must consist of linearly independent vectors.")

	# get orthonormal basis of the span using QR decomposition
	mat_from_vec = np.stack(list_of_vecs, axis=0)
	Q, _ = np.linalg.qr(mat_from_vec)
	return Q
