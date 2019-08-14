import cv2
import random
import numpy as np
import os


stride=512
image_size=512
batch_size=12



image_af=cv2.imread('./dataset/change/after/after.tif')
image_bf=cv2.imread('./dataset/change/before/before.tif')

h,w,_=image_af.shape
padding_h=(h//stride+1)*stride
padding_w=(w//stride+1)*stride
padding_img_af=np.zeros((padding_h,padding_w,3),dtype=np.uint8)
padding_img_af[0:h,0:w,:]=image_af[:,:,:]
padding_img_bf=np.zeros((padding_h,padding_w,3),dtype=np.uint8)
padding_img_bf[0:h,0:w,:]=image_bf[:,:,:]

def crop():
        n=0
        image_af=np.asarray(padding_img_af,'f')
        image_bf=np.asarray(padding_img_bf,'f')

        for i in range(padding_h//stride):
           for j in range(padding_w//stride):
              crop_af=image_af[i*stride:i*stride+image_size,j*stride:j*stride+image_size,:]
              crop_bf=image_bf[i*stride:i*stride+image_size,j*stride:j*stride+image_size,:]
              n+=1
              cv2.imwrite('./cropaf/'+str(n)+'.png',crop_af)
              cv2.imwrite('./cropbf/'+str(n)+'.png',crop_bf)

              
        if n%batch_size == 0:
              print('finished!')
        else:
           add_img_num=batch_size-(n%batch_size)
           for k in range(add_img_num):
                 add_img=np.zeros((image_size,image_size,3),dtype=np.uint8)
                 add_img=np.asarray(add_img,'f')
                 cv2.imwrite('./cropaf/'+str(n+k+1)+'.png',add_img)
                 cv2.imwrite('./cropbf/'+str(n+k+1)+'.png',add_img)
                 
           print('finished!')
if __name__ == '__main__':

    crop()
