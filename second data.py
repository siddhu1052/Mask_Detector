from tensorflow.keras.preprocessing.image import img_to_array
import os
import cv2
from tensorflow.keras.preprocessing.image import load_img
img = cv2.imread('C:\\Users\\rpnih\\Desktop\\code\\coding\\project python\\img2.jpg')
#print (img.shape)#(height,width,colour channel)
#print (img.size)
#print(img[0])
import matplotlib.pyplot as plt
plt.imshow(img)
#cv2.imshow('result',img)
while True :
	cv2.imshow('result',img)
	if cv2.waitKey(2)==27 :
		break
cv2.destroyAllWindows()
haar_data = cv2.CascadeClassifier("C:\\Users\\rpnih\\Desktop\\code\\coding\\project python\\data1.xml")
#haar_data.detectMultiScale(img)#(x,y,width,height)
while True :
	faces=haar_data.detectMultiScale(img)
	for x,y,w,h in faces :
		cv2.rectangle(img, (x,y), (x+w,y+h), (225,0,225), 2)
	cv2.imshow('result',img)
	if cv2.waitKey(2)==27 :
		break

cv2.destroyAllWindows()
import numpy
#capture= cv2.VideoCapture(0,cv2.CAP_DSHOW)

DIRECTORY=r"C:\Users\rpnih\Desktop\code\coding\project python\Face-Mask-Detection-master\dataset"
CATEGORIES={"with_mask","without_mask"}
data=[]
import os

path=os.path.join(DIRECTORY,"with_mask")
for img in os.listdir(path):
	img_path = os.path.join(path,img)
	image=load_img(img_path,target_size=(50,50))
	image=cv2.resize(image,(50,50))
	image=img_to_array(image)
      
	data.append(image)
	print(len(data))
	if(len(data)>=400):
		break
cv2.destroyAllWindows()
#numpy.save("with_mask.npy",data)

data=[]
path=os.path.join(DIRECTORY,"without_mask")
for img in os.listdir(path):
	img_path = os.path.join(path,img)
	image=load_img(img_path,target_size=(50,50))
	image=img_to_array(image)
	data.append(image)
	print(len(data))
	if(len(data)>=400):
		break
cv2.destroyAllWindows()
#numpy.save("without_mask.npy",data)
