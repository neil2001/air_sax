from pyexpat import model
import cv2
import mediapipe as mp
import time

class HandTracking():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.modelComplexity, 
            self.detectionConfidence, self.trackConfidence
            )
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNum=0, draw=True):
        landmarks = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]
            for idNum, lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((id,[cx,cy]))
                if draw:
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)

        return landmarks

def main():
    ptime = 0
    ctime = 0
    cap = cv2.VideoCapture(0)

    tracker = HandTracking()
    while True:
        success, img = cap.read()

        img = tracker.findHands(img)
        lmList = tracker.findPosition(img)
        # print(lmList[0])
    
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


    
if __name__ == "__main__":
    main()