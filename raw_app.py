import cv2
import mediapipe as mp
import time
import requests
from flask import Flask, Response, render_template
import numpy as np
import json
import pygame
import io
from PIL import Image

mp_drawing_styles = mp.solutions.drawing_styles
url = 'http://34.125.187.226:5000/api/test'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

chord_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
image_x, image_y = 200, 200
hand_landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=5)
hand_connection_drawing_spec = mp_drawing.DrawingSpec(thickness=10, circle_radius=10)
cap = cv2.VideoCapture(0)

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def main():
    while cap.isOpened():
        ret, image = cap.read()
        image = cv2.flip(image, 1)
        image_orig = cv2.flip(image, 1)
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results_hand = hands.process(image)
        Hand = results_hand.multi_hand_landmarks
        if not results_hand.multi_hand_world_landmarks:
            continue
        if Hand != None:
            for hand_landmarks in Hand:
                myLandmark = hand_landmarks.landmark[11]
                global cx,cy
                cx, cy = round(myLandmark.x,4), round(myLandmark.y,4)
                # print('X : ',cx)
                # print('Y : ',cy)
                # print('')

                if cx < 0.5000:
                    if cy < 0.4000:
                        # print("UP")
                        t1 = time.time()

                    # if cy >= 0.4800 and cy <= 0.5200:
                    #     print("STRUM")

                    if cy > 0.6000:
                        # print("DOWN")
                        t2 = time.time()
                        speed = t2-t1
                        print('t2 - t1 : ',speed)
                        leftOps(image,speed)
                        print('back to main function')

        image = rescale_frame(image, percent=75)
        # Draw the lines on the image
        cv2.line(image_orig, (int(image_orig.shape[1]/2), 0), (int(image_orig.shape[1]/2), image_orig.shape[0]), (0, 0, 255), 5)
        cv2.line(image_orig, (0, int(image_orig.shape[0]*0.4)), (int(image_orig.shape[1]/2), int(image_orig.shape[0]*0.4)), (255, 0, 0), 5)
        cv2.line(image_orig, (0, int(image_orig.shape[0]*0.6)), (int(image_orig.shape[1]/2), int(image_orig.shape[0]*0.6)), (0, 255, 0), 5)    
        for hand_landmarks in results_hand.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image_orig,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())    
        cv2.imshow("Play Guitar", image_orig)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    hands.close()
    cap.release()

def leftOps(image,speed):
    height, width, waste = image.shape
    left_image = image[:,width//2:]
    left_image_orig = left_image
    # cv2.imshow("Left half", left_image_orig)
    L_results_hand = hands.process(left_image)

    left_image.flags.writeable = True    
    left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR)
        
    if L_results_hand.multi_hand_landmarks:
        for hand_landmarks in L_results_hand.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=left_image,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=hand_landmark_drawing_spec,
                connection_drawing_spec=hand_connection_drawing_spec)
                
    res = cv2.bitwise_and(left_image, cv2.bitwise_not(left_image_orig))
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret, th1 = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        x1, y1, w1, h1 = cv2.boundingRect(contour)
        save_img = gray[y1:y1 + h1, x1:x1 + w1]
        save_img = cv2.resize(save_img, (image_x, image_y))
        global skeletal 
        skeletal = save_img
        cv2.imshow('skeletal left',save_img)

    _, img_encoded = cv2.imencode('.jpg', save_img)
    response = requests.post(url, data=img_encoded.tobytes(), headers=headers)
    
    pred_class = (json.loads(response.text)['message'])
    chordVal = chord_dict[pred_class]
    print(chordVal)

    print("Playing the {} chord sound".format(chordVal))

    mySpeed = "fast"
    
    if speed > 1:
        mySpeed = "slow"

    pygame.mixer.init()
    pygame.mixer.music.load("chord_sounds/{}_{}.wav".format(chordVal,mySpeed))
    pygame.mixer.music.play()

    return

main()