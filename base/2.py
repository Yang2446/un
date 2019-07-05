# 那么像提取图像的边缘就是去掉低频保留高频：先通过傅里叶正变换得到幅值谱，去掉低频像素，然后再通过傅里叶逆变换得到图像边缘。上代码：
import cv2
import numpy as np
import matplotlib.pyplot as plt
# img=cv2.imread('../images/can.jpg',0)
# img=cv2.imread('../images/qin.jpg',0)
img=cv2.imread('input.jpg',0)
# 高通滤波 numpy 用的多
# 正变换
f=np.fft.fft2(img)
fshift=np.fft.fftshift(f)
magnitude_spectrum=20*np.log(np.abs(fshift))


rows,cols=img.shape
crow,ccol=int(rows/2),int(cols/2)
print('img.shape',img.shape)
# 低频过滤
fshift[(crow-30):(crow+30),(ccol-30):(ccol+30)]=0
#逆变换
f_ishift=np.fft.ifftshift(fshift)
img_back=np.abs(np.fft.ifft2(f_ishift))


plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title('Input Image'),plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(img_back,cmap='gray'),plt.title('img_back'),plt.xticks([]),plt.yticks([])
plt.show()
