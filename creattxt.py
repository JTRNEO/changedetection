import os 
a=os.listdir('./patch/')
a.sort(key=lambda x:int(x[:-4]))
path='./sample_images_list.txt'
file=open(path,'w')
for i in a:
    msg=i+'\n'
    file.write(msg)
file.close()