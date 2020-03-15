import time

from cv2 import cv2

cap = cv2.VideoCapture(0)

FPS_CALC_INTERVAL = 10  # Seconds
last_calc_time = time.time()

frame_count = 0
time_capture_start = time.time()

while True:
    status, frame = cap.read()
    frame_count += 1

    time_now = time.time()
    if time_now - last_calc_time >= FPS_CALC_INTERVAL:
        last_calc_time = time_now
        total_capture_time = time_now - time_capture_start
        fps = frame_count / total_capture_time
        print(
            "FPS: {} | Frame count: {} | Total capture time: {}s".format(
                fps, frame_count, total_capture_time
            )
        )
