import cv2
import random
import numpy as np
import os
from keras.preprocessing.image import img_to_array
from deeplabv3 import Deeplabv3plus
from tqdm import tqdm




image_size = 512
bs=12




    
def predict():
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = Deeplabv3plus('./model/')
    stride = 512
    n=0
    a=os.listdir('./cropaf/')
    
    count=0
    while n < len(a):
        img_files=[]
        for i in range(bs):
            n+=1
            img_files.append(['./cropaf/'+str(n)+'.png','./cropbf/'+str(n)+'.png'])
        
        
        pred = model.predict(img_files)
        for item in pred:
                img=item['decoded_labels']
                
                mask=img[:,:,0]

                mask = mask.reshape((512,512)).astype(np.uint8)
                #print 'pred:',pred.shape


                count+=1  
                cv2.imwrite('./predict/'+str(count)+'.png',mask)
        
    

    
if __name__ == '__main__':
    
    predict()



