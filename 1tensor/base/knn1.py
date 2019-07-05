from numpy import *
import knn
import operator

c,l=knn.creatDate()
print c
print l
a=knn.classify0([0,0],c,l,3)
print a
