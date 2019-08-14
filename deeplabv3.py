from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import argparse
import os
import sys
import numpy as np 
import tensorflow as tf
import math
from skimage.measure import find_contours,approximate_polygon
import deeplab_model
from utils import preprocessing
from utils import dataset_util

from PIL import Image
import matplotlib.pyplot as plt

class Deeplabv3plus():
    def __init__(self, modelPath):
        self.modelPath = modelPath
        
        self.output_stride=16
        self.batch_size=1
        self.base_architecture='resnet_v2_101'
        self._NUM_CLASSES=2
    
    
    def predict(self,image_files):
        model = tf.estimator.Estimator(
      model_fn=deeplab_model.deeplabv3_plus_model_fn,
      model_dir=self.modelPath,
      params={
          'output_stride': self.output_stride,
          'batch_size': 12,  # Batch size must be 1 because the images' size may differ
          'base_architecture': self.base_architecture,
          'pre_trained_model': None,
          'batch_norm_decay': None,
          'num_classes': self._NUM_CLASSES,
      })
       
        predictions = model.predict(
        input_fn=lambda: preprocessing.eval_input_fn(image_files))
        return predictions
       

    

    def infere(self, image_files, imageId=None, debug=False):
            predictions = self.predict(image_files)
            result=[]
            for data in predictions:

              
              
              
                for i in range(13):
                        
                        label =i
                        img = data['decoded_labels']
                        img=img[:,:,0]
                       
                        if np.all(img==0):
                            continue
                        img=np.where(img==255,0,img)
                        cat_img=np.where(img==i,1,0)
                        
                        

                        mask = cv2.resize(cat_img.astype(np.uint8), (512,512))
                        area, perimetr, cv2Poly   = self.getMaskInfo(mask, (10,10))

                        if cv2Poly is None:
                            
                            print("Warning: Object is recognized, but contour is empty!")
                            continue

                        verts = cv2Poly[:,0,:]
                        r = {'classId': data['classes'][0][i],
                            
                            'label': label,
                            'area': area,
                            'perimetr': perimetr,
                            'verts': verts}

                        if imageId is not None:
                            r['objId'] = "{}_obj-{}".format(imageId, i)

                        result.append(r)

            return result

    def getMaskInfo(self, img, kernel=(10, 10)):

        #Define kernel
        kernel = np.ones(kernel, np.uint8)

        #Open to erode small patches
        thresh = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        #Close little holes
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel, iterations=4)

        thresh=thresh.astype('uint8')
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        maxArea = 0
        maxContour = None

        # Get largest area contour
        for cnt in contours:
            a = cv2.contourArea(cnt)
            if a > maxArea:
                maxArea = a
                maxContour = cnt

        if maxContour is None: return [None, None, None]

        perimeter = cv2.arcLength(maxContour,True)

        # aproximate contour with the 1% of squared perimiter accuracy
        # approx = cv2.approxPolyDP(maxContour, 0.01*math.sqrt(perimeter), True)

        return maxArea, perimeter, maxContour





