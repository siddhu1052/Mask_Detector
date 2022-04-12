import cv2
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
capture= cv2.VideoCapture(0,cv2.CAP_DSHOW)
data=[]
while True:
	flag,img = capture.read()	
	if flag :
		faces=haar_data.detectMultiScale(img)
		
		for x,y,w,h in faces :
			cv2.rectangle(img, (x,y), (x+w,y+h), (225,0,225), 2)
			face=img[y:y+h,x:x+w,:]
			face=cv2.resize(face,(50,50))
			print(len(data))
			if(len(data)<400):
				data.append(face)
		cv2.imshow('result',img)
		if cv2.waitKey(2)==27 or len(data)>=400:
			break
capture.release()
cv2.destroyAllWindows()
#numpy.save("with_mask.npy",data)