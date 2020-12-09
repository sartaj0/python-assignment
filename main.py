import pygame
import cv2
import pickle
import os
import numpy as np
from transitions import Machine
import face_recognition

pygame.init()
font = pygame.font.SysFont(None, 45)
font2 = pygame.font.SysFont(None, 30)
model = pickle.loads(open("encodings.pickle", "rb").read())
space = 1.25
size = 500

class Matter(object):
    pass

FaceRecognitionMachine = Matter()
transitions = [
	{ 'trigger': 'deactivate', 'source': 'MachineActive', 'dest': 'MachineInActive' },
    { 'trigger': 'activate', 'source': 'MachineInActive', 'dest': 'MachineActive' }
]
machine = Machine(FaceRecognitionMachine, states=['MachineActive', 'MachineInActive'], 
	transitions=transitions, initial='MachineInActive')


def resize(image, width=None, height=None):
	h, w, c = image.shape
	if (width is None) & (height is None):
	    raise Exception("Height and Width npth are None")
	elif (width is not None) & (height is not None):
	    raise Exception("You haved passed npth Height and Width both value")
	elif (width is not None) & (height is None):
	    
	    height = int((h / w) * width)
	    return cv2.resize(image, (width, height))
	elif (width is None) & (height is not None):

	    width = int((w / h) * height)
	    return cv2.resize(image, (width, height))


cap = cv2.VideoCapture(0)
ret, frame = cap.read()
#frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
frame = resize(frame, height=size)
h, w, c = frame.shape
display = pygame.display.set_mode((int(w * space), int(h * space)))
pygame.display.set_caption("Face Recognition")

clock = pygame.time.Clock()
run = True


class Recognizer():
	def __init__(self, model):
		self.data = pickle.loads(open(model, "rb").read())
		self.no, en = self.data["encodings"].shape

	def detect(self, image):
		self.image = image
		boxes = face_recognition.face_locations(self.image, model="hog")
		if (boxes != []) & (FaceRecognitionMachine.is_MachineInActive()):
			FaceRecognitionMachine.activate()
		elif (boxes == []) & (FaceRecognitionMachine.is_MachineActive()):
			FaceRecognitionMachine.deactivate()
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
			#cv2.rectangle(self.image, (left, top), (right, bottom), (255, 255, 0), 2)
			y = top - 25 if top - 25 > 25 else top + 25
			#cv2.putText(self.image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
			locations.append([top, right, bottom, left, name, y])

		return self.image, locations
recognizer = Recognizer("encodings.pickle")


while run:

	ret, frame = cap.read()
	#frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
	
	if frame is None:
		run = False
		break
	frame = resize(frame, height=size)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			run = False


	frame, locations = recognizer.detect(frame)


	frame = cv2.flip(frame, 0)
	frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
	
	surf = pygame.surfarray.make_surface(frame)
	display.blit(surf, (int(w * (space - 1)/2), int(h * (space - 1)/2)))

	if locations is not None:
		for location in locations:
			[top, right, bottom, left, name, y] = location

			wi = right - left
			he = bottom - top
			top = int(h * (space - 1)/2) + top
			left = int(w * (space - 1)/2) + left
			y = int(h * (space - 1)/2) + y

			pygame.draw.rect(display, (0, 255, 0), (left, top, wi, he), 4)
			text = font2.render(name, True, (255, 0, 0))
			display.blit(text, (left, y))


	text = "Application is InActive"
	color = (255, 0, 0)
	if FaceRecognitionMachine.is_MachineActive():
		text = "Application is Active"
		color = (0, 255, 0)


	text_s = font.render(text, True, color)
	text_rect = text_s.get_rect()
	text_rect.centerx = int((w * space) / 2)
	text_rect.y = 0
	display.blit(text_s, text_rect)

	pygame.display.update()
	display.fill((0, 0, 0))

pygame.quit()
