# -*- coding: utf-8 -*-
"""
Created on Sun May 19 19:49:09 2019

@author: Jahanifar
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

colorPallete = [(random.randrange(0, 240), random.randrange(0, 240), random.randrange(0, 240)) for i in range(1000)]#np.uint8(255*np.random.rand(1000,3))
drawing=False # true if mouse is pressed
mode=True # if True, draw rectangle. Press 'm' to toggle to curve

# mouse callback function
def begueradj_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode, color
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),colorPallete[2*ind],5)
            cv2.line(signal,(current_former_x,current_former_y),(former_x,former_y),(ind,ind,ind),1)
            current_former_x = former_x
            current_former_y = former_y
                #print former_x,former_y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),colorPallete[2*ind],5)
        current_former_x = former_x
        current_former_y = former_y
    return signal    


ind=1
im = cv2.imread("test.png")
signal = np.zeros(im.shape,dtype='uint8')
cv2.namedWindow("Bill BEGUERADJ OpenCV")
cv2.setMouseCallback('Bill BEGUERADJ OpenCV',begueradj_draw)
while(1):
    cv2.imshow('Bill BEGUERADJ OpenCV',im)
    k=cv2.waitKey(1)&0xFF
    if k==27:
        break
    elif k==ord("i"):
        ind+=1
    elif k==ord("d"):
        ind-=1
    
cv2.destroyAllWindows()