import cv2
import os
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# ---------------- SETTINGS ----------------
width, height = 1280, 720
folderPath = r"C:\Users\HP\PycharmProjects\PythonProject\presentation"

# ---------------- CAMERA SETUP ----------------
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

# ---------------- LOAD SLIDES ----------------
pathImages = sorted(os.listdir(folderPath), key=len)
print("Slides found:", pathImages)

# ---------------- VARIABLES ----------------
imgNumber = 0
hs, ws = 120, 213
gestureThreshold = int(height / 2)
buttonPressed = False
buttonCounter = 0
buttonDelay = 30
annotations = [[]]
annotationNumber = -1
annotationStart = False
xp, yp = 0, 0
smoothing = 5

# ---------------- HAND DETECTOR ----------------
detector = HandDetector(detectionCon=0.8, maxHands=1)

# ---------------- WINDOW SETTINGS ----------------
cv2.namedWindow("Presentation", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Presentation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ---------------- MAIN LOOP ----------------
while True:
    success, img = cap.read()
    if not success:
        print("Frame not captured from camera.")
        continue

    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    if imgCurrent is None:
        print("Could not load image:", pathFullImage)
        continue

    # ✅ Keep aspect ratio & center slide
    hImg, wImg, _ = imgCurrent.shape
    scale = min(width / wImg, height / hImg)
    new_w, new_h = int(wImg * scale), int(hImg * scale)
    imgCurrent = cv2.resize(imgCurrent, (new_w, new_h))

    # Create black background and center the slide
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x_offset = (width - new_w) // 2
    y_offset = (height - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = imgCurrent
    imgCurrent = canvas

    # ---------------- HANDS ----------------
    hands, img = detector.findHands(img)
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

    if hands and not buttonPressed:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand["center"]
        lmList = hand["lmList"]

        xRaw = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yRaw = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))
        xVal = int(xp + (xRaw - xp) / smoothing)
        yVal = int(yp + (yRaw - yp) / smoothing)
        indexFinger = (xVal, yVal)
        xp, yp = xVal, yVal

        # SLIDE CONTROL
        if cy <= gestureThreshold:
            if fingers == [1, 0, 0, 0, 0]:
                print("Previous Slide")
                if imgNumber > 0:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
                    imgNumber -= 1

            if fingers == [0, 0, 0, 0, 1]:
                print("Next Slide")
                if imgNumber < len(pathImages) - 1:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
                    imgNumber += 1

        # POINTER
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        # DRAW
        if fingers == [0, 1, 0, 0, 0]:
            if not annotationStart:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            annotations[annotationNumber].append(indexFinger)
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
        else:
            annotationStart = False

        # ERASE
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True

    # BUTTON PRESS DELAY
    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False

    # DRAW ANNOTATIONS
    for i in range(len(annotations)):
        for j in range(1, len(annotations[i])):
            cv2.line(imgCurrent, annotations[i][j - 1], annotations[i][j], (0, 0, 200), 12)

    # ✅ CAMERA TOP-RIGHT
    imgSmall = cv2.resize(img, (ws, hs))
    imgCurrent[0:hs, width - ws:width] = imgSmall

    # DISPLAY
    cv2.imshow("Webcam", img)
    cv2.imshow("Presentation", imgCurrent)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
