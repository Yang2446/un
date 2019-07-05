import numpy as np
class Network(object):
    def __init__(self,sizes):
        self.num_layers=len(sizes)
        self.sizes=sizes
        self.biases=[np.random.randn(y,1)for y in sizes[1:]]
        self.weights=[np.random.randn(y,x)for x,y in zip(sizes[:-1])]
sizes=[2,3,1]
bias=[np.random.randn(y,1)for y in sizes[1:]]
z=np.random.randn(2,1)
c=sizes[:-1]
print(c)
