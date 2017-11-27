import numpy as np
import cv2

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('test.avi')

# Check if camera opened successfully
if (cap.isOpened()== False): 
	print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
	print("Cap is opened")
	# Capture frame-by-frame
	ret, frame = cap.read()
	if ret == True:
 
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