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

model = load_model('guitar_learner4.h5')
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
        global image_orig
        image_orig = cv2.flip(image, 1)
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        
        global results_hand
        results_hand = hands.process(image)
        whichHand = results_hand.multi_hand_landmarks
        
        handType = ""
        if whichHand != None:
            for hand_landmarks in whichHand:
                myLandmark = hand_landmarks.landmark[11]
                cx, cy = myLandmark.x, myLandmark.y
                #print(cx, cy)

            leftOps(image)
        
    hands.close()
    cap.release()


def leftOps(image):
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results_hand.multi_hand_landmarks:
        for hand_landmarks in results_hand.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=hand_landmark_drawing_spec,
                connection_drawing_spec=hand_connection_drawing_spec)
    res = cv2.bitwise_and(image, cv2.bitwise_not(image_orig))

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret, th1 = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea)
        contour = contours[-1]
        x1, y1, w1, h1 = cv2.boundingRect(contour)
        # cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
        save_img = gray[y1:y1 + h1, x1:x1 + w1]
        save_img = cv2.resize(save_img, (image_x, image_y))
        pred_probab, pred_class = keras_predict(model, save_img)
        print('Predicted chord class : ',pred_class)
        #print('Probability of {pc} : {pp}'.format(pc=pred_class,pp=pred_probab))
        #print(pred_class, pred_probab)
                
    cv2.putText(image, str(chord_dict[pred_class]), (x1 + 50, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 0, 0), 9)
    print('Chord value : ',chord_dict[pred_class])
    #print(type(chord_dict[pred_class]))
    # data['chord_value'] = chord_dict[pred_class]
    # post_response = requests.post(url, json = data)
    print('')
    
    image = rescale_frame(image, percent=75)
    cv2.imshow("Left Hand Chord Classification", image)


def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def keras_process_image(img):
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


main()