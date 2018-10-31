import keras
import numpy as np
import pickle
import cv2
import os

current_directory = os.getcwd()
video_capture = cv2.VideoCapture(0)
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))

while (True):
    isOkay, capture = video_capture.read()

    if isOkay:
        capture = cv2.flip(capture, 1)

        input_image = cv2.resize(capture, (192, 192))
        input_image = input_image.astype('float') / 255.0
        input_image = keras.preprocessing.image.img_to_array(input_image)
        input_image = np.expand_dims(input_image, axis=0)

        model = keras.models.load_model(os.path.join(current_directory, 'MyNetwork.model'))
        MLB = pickle.loads(open(os.path.join(current_directory, 'MLB.pickle'), 'rb').read())
        probability = model.predict(input_image)[0]
        IDXs = np.argsort(probability)[::-1][:2]

        for (row, col) in enumerate(IDXs):
            label = "{} : {:.2f}%".format(MLB.classes_[col], probability[col] * 100)
            cv2.putText(capture, label, (10, (row * 30) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for (label, percentage) in zip(MLB.classes_, probability):
            print("{}: {:.2f}%".format(label, percentage * 100))

        cv2.imshow("realtime-classification", capture)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

video_capture.release()
cv2.destroyAllWindows()