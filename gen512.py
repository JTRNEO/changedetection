import cv2
import random
import os
import numpy as np
from tqdm import tqdm

img_w = 512
img_h = 512 
#path='./trainimages/'
#img_list=os.listdir(path)
#img_list.sort(key=lambda x:int(x[:-4]))

image_sets =   ['./dataset/change/after/after_train.png','./dataset/change/before/before_train.png']

def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)
    

def rotate(xb,yb,zb,angle):
    M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    zb = cv2.warpAffine(zb,M_rotate, (img_w,img_h))
    return xb,yb,zb
    
def blur(img):
    img = cv2.blur(img, (3, 3))
    return img

def add_noise(img):
    for i in range(200): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img
    
    
def data_augment(xb,yb,zb):
    if np.random.random() < 0.25:
        xb,yb ,zb= rotate(xb,yb,zb,90)
    if np.random.random() < 0.25:
        xb,yb,zb = rotate(xb,yb,zb,180)
    if np.random.random() < 0.25:
        xb,yb,zb = rotate(xb,yb,zb,270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)
        zb = cv2.flip(zb, 1)
        
    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb,1.0)
        yb =random_gamma_transform(yb,1.0)
    if np.random.random() < 0.25:
        xb = blur(xb)
        yb=blur(yb)
    
    if np.random.random() < 0.2:
        xb = add_noise(xb)
        yb=add_noise(yb)
    return xb,yb,zb

def creat_dataset(image_num =  15440):
    print('creating dataset...')
    #image_each = image_num / len(image_sets)
    
    image_each=image_num
    
    count=0
    src_img_af = cv2.imread(image_sets[0])  # 3 channels
    src_img_bf = cv2.imread(image_sets[1])
    label_img = cv2.imread('./dataset/change/changelabel/change_label.png',cv2.IMREAD_GRAYSCALE)  # single channel
    X_height,X_width,_ = src_img_af.shape
    while count < image_each:
        random_width = random.randint(0, X_width - img_w - 1)
        random_height = random.randint(0, X_height - img_h - 1)
        src_roi_af = src_img_af[random_height: random_height + img_h, random_width: random_width + img_w,:]
        src_roi_bf = src_img_bf[random_height: random_height + img_h, random_width: random_width + img_w,:]
        label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]

        if np.all(label_roi==0):
            continue
        else:
            src_roi_af,src_roi_bf,label_roi = data_augment(src_roi_af,src_roi_bf,label_roi)
            
            #concat=np.concatenate((src_roi_bf,src_roi_af),axis=2)
            #np.save('./dataset/mydata/JPEGImages/%d.npy'%count,concat)    
            
                
            cv2.imwrite(('./dataset/mydata/afJPEGImages/%d.jpg' % count),src_roi_af)
            cv2.imwrite(('./dataset/mydata/bfJPEGImages/%d.jpg'%count),src_roi_bf)
            cv2.imwrite(('./dataset/mydata/SegmentationClass/%d.png' % count),label_roi)
            count += 1 
            print(count)
                
creat_dataset()

