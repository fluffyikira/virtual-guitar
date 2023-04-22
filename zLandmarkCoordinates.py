import cv2
import mediapipe as mp

import requests
from keras.models import load_model
import numpy as np
import json

url = "http://localhost:8080"

data = {
    "chord_value" : ""
}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

def main():

    while cap.isOpened():
        ret, image = cap.read()
        image = cv2.flip(image, 1)
        image_orig = cv2.flip(image, 1)
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        
        results_hand = hands.process(image)
        whichHand = results_hand.multi_hand_landmarks
        
        if whichHand != None:
            for hand_landmarks in whichHand:
                myLandmark = hand_landmarks.landmark[11]
                cx, cy = round(myLandmark.x,4), round(myLandmark.y,4)
                if cy > 0.5200:
                    print("DOWN")
                if cy >= 0.4800 and cy <= 0.5200:
                    print("STRUM")
                if cy < 0.4800:
                    print("UP")
                print('X : ',cx)
                print('Y : ',cy)
                print('')

    hands.close()
    cap.release()

main()