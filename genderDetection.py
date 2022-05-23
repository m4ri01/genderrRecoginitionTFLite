import cv2
from tensorflow.keras.models import load_model
import numpy as np
import time,sys
model = load_model("gender2.h5")
labels_dict={1:'male',0:'female'}
color_dict={1:(0,0,255),0:(255,0,0)}

size = 4
webcam = cv2.VideoCapture(0) #ganti dengan port kamera (0/1/2)

# We load the xml file
classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
prev_frame_time = 0
new_frame_time = 0

while True:
    try:
        (rval, im) = webcam.read()
        mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
        faces = classifier.detectMultiScale(mini)

        for f in faces:
            (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
            face_img = im[y:y+h, x:x+w]
            resized=cv2.resize(face_img,(224,224))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,224,224,3))
            reshaped = np.vstack([reshaped])
            result=model.predict(reshaped)
            label=np.argmax(result,axis=1)[0]
            cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
            cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
            cv2.putText(im, "{}".format(labels_dict[label]), (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv2.putText(im,"FPS = {}".format(fps), (30,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2) 
        cv2.imshow('gender',   im)
        key = cv2.waitKey(1)
        if key == 27:
            break
    except Exception as e:
        print(e)
webcam.release()
cv2.destroyAllWindows()
