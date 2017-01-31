# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 07:14:02 2017

@author: uidg5371
"""

import cv2
import pickle
from threshold import threshold
import matplotlib.pyplot as plt
import numpy as np
from line import Line
from visu import show_all
from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
# %matplotlib qt

# Load calibration data
calib = pickle.load( open( "camera_cal\\calib.p", "rb" ) )
mtx   = calib["mtx"]
dist  = calib["dist"]

MAX_AGE_PREDICTED = 5
    
def perspective_transform(image):
    """Warp image according to perspective transformation"""
    
    # Get image shape
    shape = (image.shape[0], image.shape[1]) 
    
    # Define source points for perspective transformation
    x_top_right = int(0.6*shape[1])
    x_top_left  = int(0.4*shape[1])
    y_top       = int(shape[0]/1.5)
    src         = np.float32([[0,shape[0]],[shape[1],shape[0]],[x_top_right,y_top], [x_top_left,y_top]])
    
    # Define destiation points to undo perspective transformatin
    offset      = 100 # offset for dst points
    dst         = np.float32([[offset,shape[0]-offset],[shape[1]-offset,shape[0]-offset],[shape[1]-offset,offset],[offset,offset]])
    
#    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    
    return warped

def get_inv_perspective_transform(image):
    shape = (image.shape[0], image.shape[1]) 
    
    x_top_right = int(0.6*shape[1])
    x_top_left  = int(0.4*shape[1])
    y_top       = int(shape[0]/1.5)
    src         = np.float32([[0,shape[0]],[shape[1],shape[0]],[x_top_right,y_top], [x_top_left,y_top]])
    offset      = 100 # offset for dst points
    dst         = np.float32([[offset,shape[0]-offset],[shape[1]-offset,shape[0]-offset],[shape[1]-offset,offset],[offset,offset]])
    
#    # Given src and dst points, calculate the perspective transform matrix
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    return Minv
    
def find_lanes_histogram(warped, left_lane, right_lane):
    print('Try to find lanes by histogram approach!')
    # Increase prediction age
    left_lane.reset_frame()  
    right_lane.reset_frame()
    
    histogram      = np.sum(warped[int(warped.shape[0]/2):,:], axis=0)
    left_lane_idx  = np.argmax(histogram[:int(histogram.shape[0]/2)])
    right_lane_idx = np.argmax(histogram[int(histogram.shape[0]/2):]) + int(histogram.shape[0]/2)
    
    search_boxes_left,  left_detected  = left_lane.detect_from_scratch(warped, left_lane_idx)
    search_boxes_right, right_detected = right_lane.detect_from_scratch(warped, right_lane_idx)
    
    if left_detected == False:
        left_lane.detected = False
    if right_detected == False:
        right_lane.detected = False
    
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((warped, warped, warped))*255
    out_img[left_lane.ally, left_lane.allx]   = [255, 0, 0]
    out_img[right_lane.ally, right_lane.allx] = [0, 0, 255]
    
    # Show search boxes
    for box in search_boxes_left+search_boxes_right:
        cv2.rectangle(out_img,(box[0],box[1]),(box[2],box[3]),(0,255,0), 2) 
    
    if left_detected:
        left_lane.update_params(warped.shape[0])
    else:
        left_lane.detected = False
        print('Left lane not detected!')
        
    if right_detected:
        right_lane.update_params(warped.shape[0])
    else:
        right_lane.detected = False
        print('Right lane not detected!')
        
    if left_lane.age_predicted < MAX_AGE_PREDICTED and right_lane.age_predicted < MAX_AGE_PREDICTED:
        # Generate x and y values for plotting
        fity        = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        left_fitx   = left_lane.best_fit_image[0]*fity**2 + left_lane.best_fit_image[1]*fity + left_lane.best_fit_image[2]
        right_fitx  = right_lane.best_fit_image[0]*fity**2 + right_lane.best_fit_image[1]*fity + right_lane.best_fit_image[2]
    
        out_img[fity.astype(int), left_fitx.astype(int)]  = [255, 255, 0]
        out_img[fity.astype(int), right_fitx.astype(int)] = [255, 255, 0]
        
    if left_detected and right_detected:
        left_ok = left_lane.check_sanity(right_lane)
        if not left_ok:
            print('Left lane not plausible!')
        right_ok = right_lane.check_sanity(left_lane)
        if not right_ok:
            print('Right lane not plausible!')
    else:
        left_lane.detected  = False
        right_lane.detected = False
        
    return out_img

def find_lanes_filter(warped, left_lane, right_lane):
    print('Try to find lanes by filter approach!')
    # Increase prediction age
    left_lane.reset_frame()  
    right_lane.reset_frame()
    
    left_detected  = left_lane.detect_from_filter(warped)
    right_detected = right_lane.detect_from_filter(warped)
    
    if left_detected:
        left_lane.update_params(warped.shape[0])
    else:
        left_lane.detected = False
        print('Left lane not detected from filter!')
        
    if right_detected:
        right_lane.update_params(warped.shape[0])
    else:
        right_lane.detected = False
        print('Right lane not detected from filter!')
        
    # Visu stuff
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((warped, warped, warped))*255
    window_img = np.zeros_like(out_img)
    
    if left_lane.age_predicted < MAX_AGE_PREDICTED and right_lane.age_predicted < MAX_AGE_PREDICTED:
        # Color in left and right line pixels
       out_img[left_lane.ally, left_lane.allx]   = [255, 0, 0]
       out_img[right_lane.ally, right_lane.allx] = [0, 0, 255]
       
       # Generate a polygon to illustrate the search window area
       # And recast the x and y points into usable format for cv2.fillPoly()
       margin = 100
       fity = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
       fit_leftx = left_lane.best_fit_image[0]*fity**2 + left_lane.best_fit_image[1]*fity + left_lane.best_fit_image[2]
       left_line_window1 = np.array([np.transpose(np.vstack([fit_leftx-margin, fity]))])
       left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_leftx+margin, fity])))])
       left_line_pts = np.hstack((left_line_window1, left_line_window2))
       fit_rightx  = right_lane.best_fit_image[0]*fity**2 + right_lane.best_fit_image[1]*fity + right_lane.best_fit_image[2]
       right_line_window1 = np.array([np.transpose(np.vstack([fit_rightx-margin, fity]))])
       right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_rightx+margin, fity])))])
       right_line_pts = np.hstack((right_line_window1, right_line_window2))
        
       # Draw the lane onto the warped blank image
       cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
       cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
       out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
    if left_detected and right_detected:
        left_ok = left_lane.check_sanity(right_lane)
        if not left_ok:
            print('Left lane not plausible!')
            left_lane.detected  = False
            right_lane.detected = False
        right_ok = right_lane.check_sanity(left_lane)
        if not right_ok:
            print('Right lane not plausible!')
            left_lane.detected  = False
            right_lane.detected = False
    else:
        left_lane.detected  = False
        right_lane.detected = False
        
    return out_img
  
def get_lanes_current(warped, undist, Minv, left_lane, right_lane):      
    
    result = undist
    
    if left_lane.detected and right_lane.detected:
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        #yvals = np.asarray(left_lane.ally)
        yvals  = np.linspace(300,600,301)
        #xvals = np.asarray(left_lane.allx)
        left_fitx = left_lane.image_fit[0]*yvals**2 + left_lane.image_fit[1]*yvals + left_lane.image_fit[2] 
        pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    
        #yvals = np.asarray(right_lane.ally)
        right_fitx = right_lane.image_fit[0]*yvals**2 + right_lane.image_fit[1]*yvals + right_lane.image_fit[2] 
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
        
        pts = np.hstack((pts_left, pts_right))
    
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        #plt.imshow(result)
        #plt.show()
        
    return result
    
def get_lanes_best(warped, undist, Minv, left_lane, right_lane):      
    
    result = undist
    
    if left_lane.age_predicted < MAX_AGE_PREDICTED and right_lane.age_predicted < MAX_AGE_PREDICTED:
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        #yvals = np.asarray(left_lane.ally)
        yvals  = np.linspace(300,600,301)
        #xvals = np.asarray(left_lane.allx)
        left_fitx = left_lane.best_fit_image[0]*yvals**2 + left_lane.best_fit_image[1]*yvals + left_lane.best_fit_image[2] 
        pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    
        #yvals = np.asarray(right_lane.ally)
        right_fitx = right_lane.best_fit_image[0]*yvals**2 + right_lane.best_fit_image[1]*yvals + right_lane.best_fit_image[2] 
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
        
        pts = np.hstack((pts_left, pts_right))
    
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        #plt.imshow(result)
        #plt.show()
        
    return result
    
def get_point_image(warped, left_lane, right_lane):
    left_lane_img = np.zeros_like(warped).astype('uint8')
    left_lane_img[left_lane.ally, left_lane.allx] = 255
    right_lane_img = np.zeros_like(warped).astype('uint8')
    right_lane_img[right_lane.ally, right_lane.allx] = 255
    color_binary = np.dstack((np.zeros_like(warped).astype('uint8'), left_lane_img, right_lane_img))
    return color_binary

def lane_detection(image, left_lane, right_lane):
    
    global frame_cnt
    
    # Read in test image
    #image = mpimg.imread('test_images\\test6.jpg')
    
    # Camera calibration -> done in calib.py
    
    #image = gaussian_blur(image)
    
    # Distortion correction
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    
    # Color & Gradient threshold
    sobelx_thresh, color_thresh, thresholded = threshold(undist)
    
    # Perspective transform 
    warped = perspective_transform(thresholded)
    
    # Find lanes
    
    #Line finding
    # https://github.com/thomasantony/CarND-P04-Advanced-Lane-Lines/blob/7aba45bef76da4fe0c81ec9ac5622625c08367f4/Project04.ipynb
    print('pred age left: {} right:{}'.format(left_lane.age_predicted, right_lane.age_predicted))
    if (left_lane.age_predicted < MAX_AGE_PREDICTED) and (right_lane.age_predicted < MAX_AGE_PREDICTED):
        fit_img = find_lanes_filter(warped, left_lane, right_lane)
        lane_image = get_lanes_best(warped, undist, get_inv_perspective_transform(image), left_lane, right_lane)
    else:
        fit_img = find_lanes_histogram(warped, left_lane, right_lane)
        lane_image = get_lanes_current(warped, undist, get_inv_perspective_transform(image), left_lane, right_lane)
    
    # Setup visualizations
    #point_image = get_point_image(warped, left_lane, right_lane)
    
    rad_left = 10000
    if left_lane.radius_of_curvature is not None:
        rad_left = left_lane.radius_of_curvature
    rad_right = 10000
    if right_lane.radius_of_curvature is not None:
        rad_right = right_lane.radius_of_curvature
        
    rad_curvature = 0.5*(rad_left + rad_right)
    
    dist_to_center = 0
    if left_lane.age_predicted < MAX_AGE_PREDICTED and right_lane.age_predicted < MAX_AGE_PREDICTED:
        dist_to_center = (left_lane.line_base_pos-right_lane.line_base_pos)*0.5
        
    #all_img = show_all(lane_image, image, undist, thresholded, warped, point_image, sobelx_thresh, color_thresh, fit_img, rad_curvature,dist_to_center, frame_cnt)
    frame_cnt += 1
    
    return lane_image
  
left_lane      = Line()
right_lane     = Line()
frame_cnt      = 1
#lane_detection(left_lane, right_lane)

def process(image):
    global left_lane
    global right_lane
    out_img = lane_detection(image, left_lane, right_lane)
    
    return out_img
    
#val = process(mpimg.imread('test_images\\test6.jpg'))
#print(val.shape)
print('Processing video ...')
clip = VideoFileClip('challenge_video.mp4')
vid_clip = clip.fl_image(process)
out_file = 'challenge.mp4'
vid_clip.write_videofile(out_file, audio=False)