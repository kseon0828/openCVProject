import cv2
import numpy as np


#Yolo 불러오기
#Yolo 설치 필요
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
	classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#이미지 불러오기
img =cv2.imread("sample2.jps")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

#물체 감지
blob = cv2.dnn.blobFromimage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

#물체 인식 정보를 화면에 표시
class_ids = []
confidences = []
boxes = []
for out in outs:
	for detection in out:
		scores = detection[5:]
		class_id = np.argmax(scores)
		confidence = scores[class_id]
		if confidence > 0.5:
			center_x = int(detection[0] +width)
			center_y = int(detection[1] + height)
			w = int(detection[2] +width)
			h = int(detection[3] + height)

			x = int(center_x -w/2)
			y = int(center_y -h/2)
			boxes.append([x, y, w, h])
			confidence.append(float(confidence))
			class_ids.append(class_id)

#노이즈 제거
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#결과 이미지 출력
font = cv2.FONT_HERSHEV_PLAIN
for i in range(len(boxes)):
	if i in indexes:
		x, y, w, h = boxes[i]
		label = str(classes[class_ids[i]])
		color = color[i]
		cv2.rextangle(img, (x,y), (x+w, y+h), color, 2)
		cv2.putText(img, label, (x,y+30), font, 3, color, 3)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindow()