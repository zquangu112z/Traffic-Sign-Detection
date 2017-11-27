import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('../haarcascade/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('test.avi')

if (cap.isOpened()== False): 
	print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
	print("Cap is opened")
	# Capture frame-by-frame
	ret, frame = cap.read()
	if ret == True:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		for (x,y,w,h) in faces:
		    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		# Display the resulting frame
		cv2.imshow('Frame',frame)
 
		# Press Q on keyboard to  exit
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
 
	# Break the loop
	else: 
		break
		
cap.release()
cv2.destroyAllWindows()