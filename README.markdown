# realtime object recognition application with keras

### notification
- `bounding box` feature not implemented

### python version
- 3.6

### dependencies
- python packages
	- tensorflow-gpu
	- keras-gpu
	- opencv
	- imutils
	- pickle
- Y.O.L.O. version 3
	- [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
	( `yolov3.cfg` & `yolov3.txt` are included in project )

### file list
- included
	- `save-capture.py`
		- showing current capture image from webcam
		- making `capture.avi` in project directory from webcam
	- `make-dataset.py`
		- making `dataset` folder and its sub-folder for each object
	- `yolov3.cfg`
		- contains configurations for Y.O.L.O.
	- `yolov3.txt`
		- contains labels for Y.O.L.O.
	- `train-model.py`
		- making `MyNetwork.model` , `MLB.pickle` and `plot.png`
	- `MyNetwork.py`
		- contains A.N.N. for realtime classification
	- `realtime-classification.py`
		- showing the hypothesis percentage of classification on current capture image
- excluded
	- `yolov3.weights`
		- contains weight values for Y.O.L.O.

### how to setup
- install python and its packages
- prepare the webcam
- download `yolov3.weights` and locate it into project directory

### how to use
1. run `save-capture.py`
	- video recording starts with running the program
	- you can stop the recording by pressing `q`
	- output video name is `capture.avi`
2. run `make-dataset.py`
	- making dataset from `capture.avi`
	- after running the program, you can see the folder its name is `dataset` in project directory
	- `dataset` folder contains sub-folders for each object and each sub-folder contains their image(s)
3. run `train-model.py`
	- making a model for classification with the graph of `MyNetwork.py`
	- output model name is `MyNetwork.model`
	- after running the program, you can see train process at `plot.png`
4. run `realtime-classification.py`
	- showing current capture image and label name with hypothesis percentage
	- you can stop the classification by pressing `q`

### reference
- opencv video capture & file I/O
	- https://thecodacus.com/opencv-face-recognition-python-part1/#.W9px75MzaUl
- Y.O.L.O. usage with opencv
	- https://www.arunponnusamy.com/yolo-object-detection-opencv-python.html
- multi-label classification with keras
	- https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-kerasi/
