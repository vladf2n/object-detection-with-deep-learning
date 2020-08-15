import numpy as np
import time
import cv2
import os


def draw_boxes(frame, best_boxes_index_list, boxes, confidences, classIDs):
	
	if len(best_boxes_index_list) > 0:
		for i in best_boxes_index_list.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	
	return frame


def format_box_coordinates(frame_width, frame_height):
	
	box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
	(centerX, centerY, box_width, box_height) = box.astype("int")

	# Use the center (x, y) - coordinates to derive the top and and left corner of the bounding box
	x = int(centerX - (box_width / 2))
	y = int(centerY - (box_height / 2))

	# Get the bounding box coordinates, confidences, and class IDs
	box_coordinates = [x, y, int(box_width), int(box_height)]
	return box_coordinates


def feed_yolo(frame, yolo, ln):
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	yolo.setInput(blob)
	layer_outputs = yolo.forward(ln)
	return layer_outputs


def resize_image(image, scale_percent=80):

    # Calculate the 80 percent of original dimensions
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)

    size = (width, height)

    result_image = cv2.resize(image, size)

    return result_image

# Minimum probability to filter weak detections
filter_weak_detections = 0.5

# Threshold when applyong non-maxima suppression
non_maxima_suppression_threshold = 0.3

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# Initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weights_path = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configuration_path = os.path.sep.join(["yolo-coco", "yolov3.cfg"])

# Load YOLO object detector
yolo = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)

# Get layers
ln = yolo.getLayerNames()
ln = [ln[i[0] - 1] for i in yolo.getUnconnectedOutLayers()]

video = cv2.VideoCapture('videos/airport.mp4')

# writer = None
(frame_width, frame_height) = (None, None)


while True:
	
	(grabbed, frame) = video.read()
	frame = resize_image(frame)

	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if frame_width is None or frame_height is None:
		(frame_height, frame_width) = frame.shape[:2]

	# Forward pass of the YOLO object detector, giving us our bounding boxes and probabilites
	layer_outputs = feed_yolo(frame, yolo, ln)

	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layer_outputs:
		for detection in output:
			
			# Extract the class ID and confidence of the current object detection 
			current_scores = detection[5:]
			current_classID = np.argmax(current_scores)
			current_confidence = current_scores[current_classID]

			# Filter out weak predictions 
			if current_confidence > filter_weak_detections:
				
				# Get the bounding box coordinates, confidences, and class IDs
				boxes.append(format_box_coordinates(frame_width, frame_height))
				confidences.append(float(current_confidence))
				classIDs.append(current_classID)

	# Apply non-maxima suppression to suppress weak, overlapping bounding boxes
	best_boxes_index_list = cv2.dnn.NMSBoxes(boxes, confidences, filter_weak_detections, non_maxima_suppression_threshold)
	
	# Draw information on frame
	frame = draw_boxes(frame, best_boxes_index_list, boxes, confidences, classIDs)

	cv2.imshow("Detection", frame)

	key = cv2.waitKey(1)
	esc = 27
	if key == esc:
		break

video.release()