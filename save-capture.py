import cv2
import os

current_directory = os.getcwd()
video_capture = cv2.VideoCapture(0)
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
video_out = cv2.VideoWriter(os.path.join(current_directory, 'capture.avi'), cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10, (frame_width, frame_height))

while(True):
    isOkay, capture = video_capture.read()
    
    if isOkay:
        capture = cv2.flip(capture, 1)
        cv2.imshow("save-capture", capture)
        video_out.write(capture)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

video_capture.release()
video_out.release()
cv2.destroyAllWindows()