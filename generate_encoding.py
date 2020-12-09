import face_recognition
import pickle
import cv2
import os
import numpy as np

imagePaths = []
dirs = "images"
for classes in os.listdir(dirs):
	for img in os.listdir(os.path.join(dirs, classes)):
		imagePaths.append(os.path.sep.join([dirs, classes, img]))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]
	# load the input image and convert it from BGR (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb, model="hog") # hog or cnn
	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)
	
	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodings
		knownEncodings.append(encoding)
		knownNames.append(name)
		
print("[INFO] serializing encodings...")
knownEncodings = np.array(knownEncodings)
print(knownEncodings.shape)
data = {"encodings": knownEncodings, "names": knownNames}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()