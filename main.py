
import cv2
import pyttsx3

text_speech = pyttsx3.init()

import speech_recognition as sr
import pyaudio

init_rec = sr.Recognizer()
text_speech.say("Speak")
text_speech.runAndWait()
with sr.Microphone() as source:
    audio_data = init_rec.record(source, duration=5)

    text = "No value"
    text = init_rec.recognize_google(audio_data)
    text_speech.say("Input")
    text_speech.runAndWait()
    text_speech.say(text)
    text_speech.runAndWait()

thres = 0.65 # Threshold to detect object

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

classNames = []
classFile = 'coco.names'

with open(classFile,'rt') as f:
    classNames=[line.rstrip() for line in f]

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    print(classIds,bbox)

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+3),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            ans = classNames[classId-1]
            if ans == text:
               text_speech.say(ans)
               text_speech.runAndWait()
    cv2.imshow("Output",img)
    cv2.waitKey(1)
