#!/usr/bin/python3
import cv2
import numpy as np

# train and img the classifier
face_cascade=cv2.CascadeClassifier('facee.xml')
eye_cascade=cv2.CascadeClassifier('eye.xml')
glass_cascade=cv2.CascadeClassifier('glasses_cascade.xml')



cap=cv2.VideoCapture(0)
status,frame=cap.read()


while True:
	status,frame=cap.read()
	# coverting colored image into gray scale	
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	#now will return a rectangle with coordinates(x,y,w,h) around the detected face
	faces=face_cascade.detectMultiScale(gray,1.3,5)
	
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
		# region of interest
		roi_gray=gray[y:y+h,x:x+w]
		roi_color=frame[y:y+h,x:x+w]

		eyes=eye_cascade.detectMultiScale(roi_gray,1.3,5)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
		
		
		
		glass=glass_cascade.detectMultiScale(roi_gray,1.04,5)
		
		for (gx,gy,gw,gh) in glass:
			font=cv2.FONT_HERSHEY_SIMPLEX
			cv2.rectangle(roi_color,(gx,gy),(gx+gw,gy+gh),(255,255,0),2)
			cv2.putText(roi_color,'glasses',(gx,gy-3), font, 0.5, (11,255,255))
	cv2.imshow('face',frame)
	k= cv2.waitKey(30) & 0xff
	if k ==27:
		break



cap.release()
cv2.destroyAllWindows()


