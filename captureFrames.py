# Capture imagens from woulds in houl
# this algorithm saves every 5 multiple frame and change the folder if you press 's' from the keyboard
# 
# Author: Fellipe Costa
#

import cv2
import os
import numpy as np

left = cv2.VideoCapture(1)
left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
right = cv2.VideoCapture(2)
right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))


multiplier = 5
frame_count = 0
pasta = 0
str_pasta= str(pasta)

#name of the camera, the pacient and the region
camera = "svproMariaAparecidaSacral"
path = "./imagens/"+camera+str_pasta

while(True):
    
    if not (left.grab() and right.grab()):
        print("No more frames")
        break
    
    frame_count += 1
    
    _, leftFrame = left.retrieve()
    _, rightFrame = right.retrieve()

    cv2.imshow('left', leftFrame)
    cv2.imshow('right', rightFrame)

    if frame_count % multiplier == 0:
        if not (os.path.exists(path)):
            os.mkdir(path)
            os.mkdir(path+"/right")
            os.mkdir(path+"/left")

        cv2.imwrite(  path+"/right/rframe%d.jpg" % frame_count, rightFrame) #write specific frames 
        cv2.imwrite( path+"/left/lframe%d.jpg" % frame_count, leftFrame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        pasta = pasta+1
        str_pasta= str(pasta)
        path = "./imagens/"+camera+str_pasta
        print(path)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

left.release()
right.release()
cv2.destroyAllWindows()
