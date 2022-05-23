import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

webcam = cv2.VideoCapture(0)
imW = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
imH = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))

color_dict={1:(0,0,255),0:(255,0,0)}
labels_dict={1:'male',0:'female'}
classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
prev_frame_time = 0
new_frame_time = 0

GRAPH_NAME = "gender2.tflite"
LABELMAP_NAME = "labels.txt"
min_conf_threshold = float(0.5)


pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH,LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

frame_rate_calc = 1
freq = cv2.getTickFrequency()

while True:
    _,frame1 = webcam.read()
    faces = classifier.detectMultiScale(frame1)
    # boundingBoxes = []
    # classesObject = []
    for f in faces:
        (x, y, w, h) = [v for v in f]
        frame = frame1.copy()
        frame = frame[y:y+h, x:x+w]
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        # cv2.imshow('Object detector', frame_resized)
        input_data = np.expand_dims(frame_resized, axis=0)
        # print(input_data)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
        
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()
        # print(interpreter.get_tensor(output_details[0]))
        classes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = np.argmax(classes)
        # boundingBoxes.append([x,y,w,h])
        # classesObject.append(classes)
        cv2.rectangle(frame1,(x,y),(x+w,y+h),color_dict[classes],2)
        cv2.rectangle(frame1,(x,y-40),(x+w,y),color_dict[classes],-1)
        cv2.putText(frame1, "{}".format(labels_dict[classes]), (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    cv2.putText(frame1,"FPS = {}".format(fps), (30,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2) 
    cv2.imshow('gender',   frame1)
    if cv2.waitKey(27) == ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()