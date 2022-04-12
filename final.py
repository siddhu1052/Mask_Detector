import numpy as np
import cv2
with_mask = np.load(r'C:\Users\rpnih\Desktop\code\coding\project python\with_mask.npy')
without_mask=np.load(r'C:\Users\rpnih\Desktop\code\coding\project python\without_mask.npy')
print(with_mask.shape)
print(with_mask.size)
with_mask=with_mask.reshape(400,50*50*3)
without_mask=without_mask.reshape(400,50*50*3)
X=np.r_[with_mask,without_mask]
labels=np.zeros(X.shape[0])
names={0:'mask',1:'No_mask'}
labels[400:]=1.0
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


x_train, x_test, y_train, y_test=train_test_split(X,labels,test_size=0.25)
pca=PCA(n_components=3)
x_train=pca.fit_transform(x_train)
svm=SVC()
svm.fit(x_train,y_train)

#x_train, x_test, y_train, y_test=train_test_split(X,labels,test_size=0.20)
x_test = pca.transform(x_test)
y_pred = svm.predict(x_test)

print(accuracy_score(y_test,y_pred))

haar_data = cv2.CascadeClassifier("C:\\Users\\rpnih\\Desktop\\code\\coding\\project python\\data1.xml")
capture= cv2.VideoCapture(0,cv2.CAP_DSHOW)
data=[]
font=cv2.FONT_HERSHEY_SIMPLEX
while True:
	flag,img = capture.read()	
	if flag :
		faces=haar_data.detectMultiScale(img)
		
		for x,y,w,h in faces :
			
			face=img[y:y+h,x:x+w,:]
			face=cv2.resize(face,(50,50))
			face = face.reshape(1,-1)
			face=pca.transform(face)
			pred=svm.predict(face)
			n=names[int(pred)]
			if n=="mask" :
				cv2.rectangle(img, (x,y), (x+w,y+h), (5,220,15), 4)
				cv2.putText(img,n,(x,y),font,1,(5,220,15),2)
			else :
				cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,250), 4)
				cv2.putText(img,n,(x,y),font,1,(0,0,250),2)
			#image on which we want to put image,1/0,-,font,32 bites ,color,boldness)
		cv2.imshow('result',img)
		if cv2.waitKey(2)==27:
			break
capture.release()
cv2.destroyAllWindows()