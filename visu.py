# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 07:35:55 2017

@author: Friedrich Kenda-Erbs
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt 

def show_all(lane_image, orig_image, undist, thresholded, warped, point_image, sobelx_thresh, color_thresh, fit_img, rad_curvature, dist_to_center, frame_cnt):
    # middle panel text example
    # using cv2 for drawing text in diagnostic pipeline.
    font = cv2.FONT_HERSHEY_COMPLEX
    middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)
    cv2.putText(middlepanel, 'Estimated lane curvature: {}'.format(rad_curvature), (30, 60), font, 1, (255,0,0), 2)
    cv2.putText(middlepanel, 'Estimated Meters right of center: {}'.format(dist_to_center), (30, 90), font, 1, (255,0,0), 2)
    
    
    # assemble the screen example
    diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    if lane_image is not None:
        diagScreen[0:720, 0:1280] = lane_image
        plt.text(640, 50, 'Lane tracking')
                  
    diagScreen[0:240, 1280:1600] = cv2.resize(orig_image, (320,240), interpolation=cv2.INTER_AREA) 
    plt.text(1440, 20, 'Orig', fontsize=8)
    diagScreen[0:240, 1600:1920] = cv2.resize(undist, (320,240), interpolation=cv2.INTER_AREA)
    plt.text(1760, 20, 'Undist', fontsize=8)
    diagScreen[240:480, 1280:1600] = cv2.resize(np.asarray(np.dstack((thresholded, thresholded, thresholded)), dtype=np.uint8)*255, (320,240), interpolation=cv2.INTER_AREA)
    plt.text(1440, 260, 'Threshold', fontsize=8, color='red')
    diagScreen[240:480, 1600:1920] = cv2.resize(np.asarray(np.dstack((warped, warped, warped)), dtype=np.uint8)*255, (320,240), interpolation=cv2.INTER_AREA)
    plt.text(1760, 260, 'Warped', fontsize=8, color='red')
    
    if point_image is not None:
        diagScreen[600:1080, 1280:1920] = cv2.resize(point_image, (640,480), interpolation=cv2.INTER_AREA)
        plt.text(1600, 620, 'Points', fontsize=8, color='red')
        
    diagScreen[720:840, 0:1280] = middlepanel
    diagScreen[840:1080, 0:320] = cv2.resize(np.asarray(np.dstack((sobelx_thresh, sobelx_thresh, sobelx_thresh)), dtype=np.uint8)*255, (320,240), interpolation=cv2.INTER_AREA)
    plt.text(160, 860, 'Sobel', fontsize=8, color='red')
    diagScreen[840:1080, 320:640] = cv2.resize(np.asarray(np.dstack((color_thresh, color_thresh, color_thresh)), dtype=np.uint8)*255,  (320,240), interpolation=cv2.INTER_AREA)
    plt.text(480, 860, 'Color', fontsize=8, color='red')
    
    if fit_img is not None:
        diagScreen[840:1080, 640:960] = cv2.resize(fit_img, (320,240), interpolation=cv2.INTER_AREA)
        plt.text(800, 860, 'Fit', fontsize=8, color='red')
        
    #if right_fit_img is not None:
        #diagScreen[840:1080, 960:1280] = cv2.resize(right_fit_img, (320,240), interpolation=cv2.INTER_AREA)
    
    #cv2.imwrite('result_{}.png'.format(frame_cnt), diagScreen)
    #plt.imshow(diagScreen)
    #plt.show()
    
    return diagScreen

