import cv2
import numpy as np

net = cv2.dnn.readNetFromDarknet('Models/yolov3.cfg', 'Models/yolov3.weights')
classes = []
with open('Datasets/coco.names', 'r') as f:
    classes = f.read().strip().split("\n")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

def get_output_names(net):
    layer_names = net.getLayerNames()
    output_layers_indices = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in output_layers_indices]
    return output_layers

# Initialize an empty dictionary to store detected objects
detected_objects = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_names(net))

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.7:
                center_x, center_y, width, height = map(int, detection[:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]))
                x, y = center_x - width // 2, center_y - height // 2

                # Check if this object is already detected
                if class_id in detected_objects:
                    # Merge overlapping bounding boxes
                    x1, y1, w1, h1 = detected_objects[class_id]
                    x = min(x, x1)
                    y = min(y, y1)
                    width = max(x + width, x1 + w1) - x
                    height = max(y + height, y1 + h1) - y

                detected_objects[class_id] = (x, y, width, height)

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, classes[class_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
