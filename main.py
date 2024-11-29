import cv2
import time
import argparse
import pyautogui
import numpy as np
import mediapipe as mp
from model import prediction,CNN
import torch
num_workers=0
mp_drawing = mp.solutions.drawing_utils
drawingModule = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
'''
GLOBAL VARIABLES
'''
# Layout/FrontEnd of image
IMAGEHEIGHT = 480  
IMAGEWIDTH = 640
ROIWIDTH = 256
LEFT = int(IMAGEWIDTH/2 - ROIWIDTH/2)
RIGHT = LEFT + ROIWIDTH
TOP = int(IMAGEHEIGHT/2 - ROIWIDTH/2)
BOTTOM = TOP + ROIWIDTH
FONT = cv2.FONT_HERSHEY_SIMPLEX

gesture_dict= {0:'zero', 1:'one',2:'two' ,3:'OK',4:'five', 5:'thumbup',6:'thumbdown'}
gesture_action={}

def parse_args():
    parser = argparse.ArgumentParser(
        description="data collection")
    parser.add_argument(
        '--clas',
        default=0,
        help='which class of the gestures',
    )
    return parser.parse_args()

def get_position(pre_hand_result, hand_result):

        point = 8
        position = [hand_result.landmark[point].x ,hand_result.landmark[point].y]
        sx,sy = pyautogui.size()
        x_old,y_old = pyautogui.position()
        x = int(position[0]*sx)
        y = int(position[1]*sy)
        delta_x=x
        delta_y=y
        if pre_hand_result is not None:
            delta_x = x - pre_hand_result[0]
            delta_y = y - pre_hand_result[1]
            # Controller.prev_hand = x,y
        # delta_x = x - pre_hand_result.landmark[point].x
        # delta_y = y - pre_hand_result.landmark[point].y

        distsq = delta_x**2 + delta_y**2
        ratio = 1

        if distsq <= 25:
            ratio = 0
        elif distsq <= 900:
            ratio = 0.07 * (distsq ** (1/2))
        else:
            ratio = 2.1
        x_ , y_ = x_old + delta_x*ratio , y_old + delta_y*ratio
        return x_,y_, [x,y]

# At the start of your main script
device = torch.device('cpu')
model = CNN()
    #model = torch.load("weight/no_skeleton.pkl")
model.load_state_dict(torch.load('skeleton_demo.pkl'))
model.to(device)
model.eval()

if __name__ == '__main__':
    
    ARGS = parse_args()
    print("Activating your personal assistant:p...")

    cap= cv2.VideoCapture(0)

    while 1:
        label, img= cap.read()

        img=cv2.flip(img,1)
        text = "Press Enter to start your assistant... \n Gesture in the rectangle to make the command :p"
        y0, dy = 80, 20
        for i, line in enumerate(text.split('\n')):
            y = y0 + i*dy
            cv2.putText(img, line, (80, y ), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 2,cv2.LINE_AA)
        cv2.rectangle(img, (LEFT,TOP), (RIGHT, BOTTOM), (0,0,0), 1)
        cv2.namedWindow("Personal assistant", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Personal assistant", 1280,960)
        cv2.imshow('Personal assistant',img) 
        key=cv2.waitKey(5)
        if  key ==13:
            break
        elif key==ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
    imgs=[]
    act=9
    if key==13:
        with mp_hands.Hands(max_num_hands = 1,min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while 1:
                act=-1
                label, img= cap.read()
                img=cv2.flip(img,1)
                cv2.putText(img, f"Doing nothing now:p", (80, 80), FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(img, (LEFT,TOP), (RIGHT, BOTTOM), (0,0,0), 1)
                cv2.imshow('Personal assistant',img)
                cv2.waitKey(5)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img.flags.writeable = False
                results = hands.process(img)
                img.flags.writeable = True
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks: 
                    for hand_landmarks in results.multi_hand_landmarks:
                        # mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS) 
                        drawingModule.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                    drawingModule.DrawingSpec(color = (121, 22, 76), thickness = 2, circle_radius = 1),
                                                    drawingModule.DrawingSpec(color = (250, 44, 250), thickness = 2, circle_radius = 1))
                img = img[TOP:BOTTOM, LEFT:RIGHT]
                img = cv2.bilateralFilter(img, 5, 50, 100)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite('temp.jpg', img)
                imgs.append(img)
                time.sleep(0.5)
                if len(imgs)==5:
                    act= prediction(imgs,model)
                    imgs.clear()
                    print(f"predicted gesture {act} current len {len(imgs)}")
                    if act==4:
                        pyautogui.scroll(200)
                    elif act==6:
                        pyautogui.scroll(-200)
                        # continue
                    elif act==5:
                        pyautogui.click(button='left')
                if key==ord('q'):
            
                    break
    cap.release()               
    cv2.destroyAllWindows()
                