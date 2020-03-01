import numpy as np

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
