#!/usr/bin/python3


import cv2
# applying calling classifier
# train our img classifier
casclf=cv2.CascadeClassifier('facee.xml')
# loading face data
cap=cv2.VideoCapture(0)

while cap.isOpened():
	status,frame=cap.read()
	# now we can apply classifier in live frame
	
	#parameter at which image will 
# capture the live frame data eith data tune parameters
	face=casclf.detectMultiScale(frame,1.13,5) # classifier tuning parameter
	# print(face)
	for x,y,h,w in face:
		cv2.rectangle(frame,(x,y),(x+h,y+w),(0,0,255),2)
	cv2.imshow('face',frame)

	if cv2.waitKey(10) & 0xff == ord('q'):
		break

cv2.destroyAllWindows()
cap.release()

