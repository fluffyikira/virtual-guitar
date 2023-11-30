import cv2
import mediapipe as mp
import math
from scipy import misc
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
hand_landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=5)
hand_connection_drawing_spec = mp_drawing.DrawingSpec(thickness=10, circle_radius=10)
image_x, image_y = 200, 200

print("flag1")
# For webcam input:
cap = cv2.VideoCapture(0)
if cap.isOpened:
    print("flag4")
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    image.flags.writeable = False
    image = cv2.flip(image, 1)
    global image_orig
    image_orig = cv2.flip(image, 1)
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        # Here is How to Get All the Coordinates
        # for ids, landmrk in enumerate(hand_landmarks.landmark):
            # print(hand_landmarks.landmark[11])
            landmrk = hand_landmarks.landmark[11]
            cx, cy = round(landmrk.x, 4), round(landmrk.y, 4)
            if cx < 0.5:
              # print(cx, cy) 
              if cy < 0.4800: 
                 print("up")
              if cy >= 0.4800 and cy <=0.5200:
                print("STRUM")
                height, width, waste = image.shape
                left_image = image[:,width//2:]
                left_image_orig = left_image
                cv2.imshow("Left half", left_image)

                left_image.flags.writeable = True
                left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
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
                    contours = sorted(contours, key=cv2.contourArea)
                    contour = contours[-1]
                    x1, y1, w1, h1 = cv2.boundingRect(contour)
                    # cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
                    save_img = gray[y1:y1 + h1, x1:x1 + w1]
                    save_img = cv2.resize(save_img, (image_x, image_y))
                    cv2.imshow('skeletal left',save_img)

              if cy > 0.5200: 
                print("down")
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()