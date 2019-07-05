import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST',one_hot=False)
img=mnist.train.images[30]
plt.imshow(img.reshape((28,28)))