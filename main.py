import cv2 
import time
import numpy as np
import HandTrackingModule as htm
import math
import osascript

# Parameters
wCam, hCam = 480, 320

# Capture video
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

# Hand detector
detector = htm.handTracker()

# Volume control
# Define the volume range based on the length of the line between fingers
min_length = 50  # Minimum length of the line (pixels)
max_length = 350  # Maximum length of the line (pixels)
min_volume = 0  # Minimum volume (0%)
max_volume = 100  # Maximum volume (100%)

while True:
    success, img = cap.read()
    img = detector.handsFinder(img)
    lmList = detector.positionFinder(img, draw=False)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1,y1), 15, (0,255,0), cv2.FILLED)
        cv2.circle(img, (x2,y2), 15, (0,255,0), cv2.FILLED)
        cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 3)
        cv2.circle(img, (cx,cy), 15, (0,255,0), cv2.FILLED)
        
        length = math.hypot(x2-x1, y2-y1)
        # print(length)

        # Map the length of the line to the volume range
        volume = int(np.interp(length, [min_length, max_length], [min_volume, max_volume]))
        
        # Implement volume control using osascript
        osascript.run(f"set volume output volume {volume}")           

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (40,70), cv2.FONT_HERSHEY_PLAIN,
                2, (0,0,255), 2)

    cv2.imshow("Hand Tracker", img)
    cv2.waitKey(1)