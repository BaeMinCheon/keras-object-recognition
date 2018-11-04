import cv2
import numpy as np
import os
import shutil

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

current_directory = os.getcwd()
dataset_directory = os.path.join(current_directory, 'dataset')
if not os.path.exists(dataset_directory):
    os.makedirs(dataset_directory)
video_input = cv2.VideoCapture(os.path.join(current_directory, 'capture.avi'))

classes = None
output_directories = []
count_objects = []
with open(os.path.join(current_directory, 'yolov3.txt'), 'r') as file:
    classes = [line.strip() for line in file.readlines()]
    for c in classes:
        output_directory = os.path.join(dataset_directory, c)
        output_directories.append(output_directory)
        count_objects.append(0)
        shutil.rmtree(output_directory, ignore_errors=True)

colors = np.random.uniform(0, 255, size=(len(classes), 3))
scale = 0.001
net = cv2.dnn.readNet(os.path.join(current_directory, 'yolov3.weights'), os.path.join(current_directory, 'yolov3.cfg'))
confidence_threshold = 0.75

while(True):
    isOkay, capture = video_input.read()

    if isOkay:
        blob = cv2.dnn.blobFromImage(capture, scale, (224, 224), (0, 0, 0), True, False)
        net.setInput(blob)
        outputs = net.forward(get_output_layers(net))

        height = capture.shape[0]
        width = capture.shape[1]
        class_ids = []
        confidences = []
        boxes = []

        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_threshold:
                    x = int(detection[0] * width)
                    y = int(detection[1] * height)
                    wid = int(detection[2] * width)
                    hei = int(detection[3] * height)
                    ltx = abs(x - int(wid / 2))
                    lty = abs(y - int(hei / 2))

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([ltx, lty, wid, hei])

                    count_objects[class_id] = count_objects[class_id] + 1
                    if not os.path.exists(output_directories[class_id]):
                        os.makedirs(output_directories[class_id])
                    cv2.imwrite(os.path.join(output_directories[class_id], str(count_objects[class_id]) + '.png'), capture[lty : lty + hei, ltx : ltx + wid])

    else:
        break

video_input.release()