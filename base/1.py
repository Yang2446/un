import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread('input.jpg',0)
# 高通滤波 numpy 用的多
# 正变换
f=np.fft.fft2(img)
fshift=np.fft.fftshift(f)
magnitude_spectrum=20*np.log(np.abs(fshift))


plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title('Input Image'),plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum,cmap='gray'),plt.title('magnitude_spectrum')
plt.show()
