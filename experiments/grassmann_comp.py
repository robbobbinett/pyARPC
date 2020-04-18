import numpy as np
from pymanopt.manifolds import Grassmann

mfld = Grassmann(100, 9)
#print(dir(mfld))

for _ in range(10):
	temp = mfld.rand()
	print(np.linalg.norm(np.eye(9)-np.transpose(temp)@temp))
