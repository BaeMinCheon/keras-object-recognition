import cv2
import os

name = str(input('type name of the face : '))

current_directory = os.getcwd()
dataset_directory = os.path.join(current_directory, 'dataset')
output_directory = os.path.join(dataset_directory, name)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

video_input = cv2.VideoCapture(os.path.join(current_directory, 'capture.avi'))
detector = cv2.CascadeClassifier(os.path.join(current_directory, 'haarcascade_frontalface_default.xml'))
frame_counter = 0

while(True):
    isOkay, capture = video_input.read()

    if isOkay:
        capture_gray = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
        face = detector.detectMultiScale(capture_gray, 1.3, 5)
        for (ltx, lty, wid, hei) in face:
            frame_counter = frame_counter + 1
            cv2.rectangle(capture, (ltx, lty), (ltx + wid, lty + hei), (255, 0, 0), 2)
            cv2.imwrite(os.path.join(output_directory, str(frame_counter) + '.png'), capture_gray[lty : lty + 192, ltx : ltx + 192])
    else:
        break

video_input.release()
cv2.destroyAllWindows()