import cv2
import pyrealsense2 as rs

cap = cv2.VideoCapture(4)

while True:
    status, photo = cap.read()
    print(photo.shape)

    if not status:
        print("Camera not found")
        break

    cv2.imshow("Webcam Video Stream", photo)

    # Press Enter to exit
    if cv2.waitKey(10) == 13:
        break

cap.release()
cv2.destroyAllWindows()

