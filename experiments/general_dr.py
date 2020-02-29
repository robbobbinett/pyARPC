import numpy as np

def reduce_eigendicts(eigVals, eigVecs, d):
	new_vals = {}
	new_vecs = {}
	for key in eigVals.keys():
		vals = [(ind, val) for ind, val in enumerate(eigVals[key])]
		vecs = eigVecs[key].copy()
		vecs.sort
