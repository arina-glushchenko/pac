import cv2
import time

prev_frame = None
cap = cv2.VideoCapture(0)
contours_r = []
start_time = time.time()
color = "green"

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    key = cv2.waitKey(10) & 0xff
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if prev_frame is not None:
        frame_diff = cv2.absdiff(prev_frame, gray)
        frame_diff = cv2.Canny(frame_diff, 50, 150)
        contours_r, hierarchy = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if time.time() - start_time >= 10:
        if color == "green":
            color = "red"
        else:
            color = "green"
        start_time = time.time()

    if color == "green":
        cv2.putText(frame, "Green light", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
    else:
        cv2.putText(frame, "Red light", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        frame = cv2.drawContours(frame, contours_r, -1, (0, 0, 255), 1)
        frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)

    cv2.imshow('Frame', frame)

    prev_frame = gray.copy()

    if key == 27:  # Esc
        break

cv2.destroyAllWindows()
cap.release()
