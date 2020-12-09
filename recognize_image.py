import cv2
import pickle
import os
import numpy as np
import face_recognition

class Recognizer():
	def __init__(self, model):
		self.data = pickle.loads(open(model, "rb").read())
		self.no, en = self.data["encodings"].shape

	def detect(self, image):
		self.image = image
		boxes = face_recognition.face_locations(self.image, model="hog")
		if (boxes == []):
			return self.image, None

		encodings = face_recognition.face_encodings(self.image, boxes)
		names = []
		for encoding in encodings:
			matches = face_recognition.compare_faces(self.data["encodings"],
				encoding)
			name = "Unknown"

			if True in matches:
				matchedIdxs = [i for (i, b) in enumerate(matches) if b]
				counts = {}
				for i in matchedIdxs:
					name = self.data["names"][i]
					counts[name] = counts.get(name, 0) + 1
				name = max(counts, key=counts.get)
			
			names.append(name)

		locations = []
		for ((top, right, bottom, left), name) in zip(boxes, names):
			cv2.rectangle(self.image, (left, top), (right, bottom), (255, 255, 0), 2)
			y = top - 25 if top - 25 > 25 else top + 25
			cv2.putText(self.image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
			locations.append([top, right, bottom, left, name, y])

		return self.image, locations
recognizer = Recognizer("encodings.pickle")

image = cv2.imread("test/10.webp")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
frame, locations = recognizer.detect(image)
#frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imshow("IMAGE", frame)
cv2.waitKey(0)