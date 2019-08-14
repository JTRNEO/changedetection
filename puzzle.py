import time
import cv2
import os
import numpy as np

# Uncomment next two lines if you do not have Kube-DNS working.
# import os
# host = os.getenv("REDIS_SERVICE_HOST")

path='./dataset/change/after/after.tif'
stride=512
image_size=512

n=0

image=cv2.imread(path)
h,w,_=image.shape
padding_h=(h//stride+1)*stride
padding_w=(w//stride+1)*stride
mask_whole = np.zeros((padding_h,padding_w),dtype=np.uint8)

for i in range(padding_h//stride):
     for j in range(padding_w//stride):
            n+=1
            mask=cv2.imread('./predict/'+str(n)+'.png')
            mask=mask[:,:,0]
            mask=mask.reshape((image_size,image_size)).astype(np.uint8)
            mask_whole[i*stride:i*stride+image_size,j*stride:j*stride+image_size] = mask[:,:]

cv2.imwrite('./pre.tif',mask_whole[0:h,0:w])
