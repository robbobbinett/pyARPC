import numpy as np
import cvxpy as cp

def find_mveh(points):
	"""
	Solve the minimum-volume enclosing hypersphere problem
	as a quadratically constrained linear program.

	points -> d x n NumPy array, where d is the dimension
	of the ambient space and n is the number of points
	"""
	# get dimension and number of points
	d, n = points.shape

	# require that n = d + 1
	# (i.e. the points are the vertices of a simplex)
	if d != n-1:
		raise ValueError("The number of points should be equal to the dimension of the points plus one, but:\nd = "+str(d)+"\nn = "+str(n))

	# define cvxpy variables
	# rho is the square of the radius of the hypersphere
	# note that we force rho to be positive
	rho = cp.Variable(1, pos=True)
	# c is the center of the hypersphere
	c = cp.Variable(d)

	# define objective
	objective = cp.Minimize(rho)

	# construct d x d identity
	aye = np.eye(d)

	# define constraints
	constraints = []
	for j in range(n):
		constraints.append(cp.quad_form(c-points[:, j], aye))

	prob = cp.Problem(objective, constraints)
	return prob.solve()
