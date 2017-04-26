from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")

args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

cap = cv2.VideoCapture(0)

past_values_x = []
def min_intensity_x(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	min_sum_y = 255 * len(img)
	min_index_x = -1
	
	for x in range(len(img[0])):
		
		temp_sum_y = 0
		
		for y in range(len(img)):
			temp_sum_y += img[y][x]
		
		if temp_sum_y < min_sum_y:
			min_sum_y = temp_sum_y
			min_index_x = x
	
	past_values_x.append(min_index_x)
	
	if len(past_values_x) > 3:
		past_values_x.pop(0)
	
	return int(sum(past_values_x) / len(past_values_x))

past_values_y = []
def min_intensity_y(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	min_sum_x = 255 * len(img[0])
	min_index_y = -1
	
	for y in range(len(img)):
		
		temp_sum_x = 0
		
		for x in range(len(img[0])):
			temp_sum_x += img[y][x]
		
		if temp_sum_x < min_sum_x:
			min_sum_x = temp_sum_x
			min_index_y = y
	
	past_values_y.append(min_index_y)
	
	if len(past_values_y) > 3:
		past_values_y.pop(0)
	
	return int(sum(past_values_y) / len(past_values_y))

def extract_eye(image, left, bottom_left, bottom_right, right, upper_right, upper_left):
	lower_bound = max([left[1], right[1], bottom_left[1], bottom_right[1], upper_left[1], upper_right[1]])
	upper_bound = min([left[1], right[1], upper_left[1], upper_right[1], bottom_left[1], bottom_right[1]])

	eye = image[upper_bound-3:lower_bound+3, left[0]-3:right[0]+3]
	
	pupil_x = min_intensity_x(eye)
	pupil_y = min_intensity_y(eye)
	
	cv2.line(eye,(pupil_x,0),(pupil_x,len(eye)),(0,255,0), 1)
	cv2.line(eye,(0,pupil_y),(len(eye[0]),pupil_y),(0,255,0), 1)
	
	cv2.line(image,((bottom_left[0] + bottom_right[0]) / 2, lower_bound), ((upper_left[0] + upper_right[0]) / 2, upper_bound),(0,0,255), 1)
	cv2.line(image,(left[0], left[1]), (right[0], right[1]),(0,0,255), 1)
	
	image[upper_bound-3:lower_bound+3, left[0]-3:right[0]+3] = eye
	return eye
	
while(True):
	# load the input image, resize it, and convert it to grayscale
	ret, image = cap.read()
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# detect faces in the grayscale image
	rects = detector(gray, 1)
	
	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
	
		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		#(x, y, w, h) = face_utils.rect_to_bb(rect)
		#cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	
		# show the face number
		#cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		#	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		count = 1
		right_eye = imutils.resize(extract_eye(image, shape[36], shape[41], shape[40], shape[39], shape[38], shape[37]), width=100, height=50)
		
		for (x, y) in shape:
			if count > 36 and count < 43:
					cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
				
			count += 1

		image[0:len(right_eye),0:len(right_eye[0])] = right_eye
	cv2.imshow("PupilTrack v.0.1", image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
# show the output image with the face detections + facial landmarks

cv2.waitKey(0)