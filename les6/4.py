import cv2

cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break
    video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('video', video)

    key = cv2.waitKey(20) & 0xFF
    if key == 27:  # Esc
        break

cv2.destroyWindow('video')
cap.release()