# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 07:21:43 2017

@author: Friedrich Kenda-Erbs
"""

import numpy as np
#import cv2

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        #self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        #self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None 
        #polynomial coefficients averaged over the last n iterations for fit in image space
        self.best_fit_image = None 
        #polynomial coefficients for the most recent fit for world coordinates
        self.current_fit = [np.array([False])]  
        #fit coefficients for the most recent fit for image coordinates
        self.image_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = []  
        #y values for detected line pixels
        self.ally = []
        # number of frames lane is predicted
        self.age_predicted = 10000
        
    def reset_frame(self):
        self.age_predicted       = self.age_predicted + 1
        self.radius_of_curvature = None
        self.line_base_pos       = None 
        self.diffs               = np.array([0,0,0], dtype='float') 
        
    def check_sanity(self, other_line):     
        
        curvature_own   = 1./self.radius_of_curvature
        curvature_other = 1./other_line.radius_of_curvature
        diff_curvature = abs(curvature_own - curvature_other)
        radius_similar = diff_curvature < 0.001
        
        dist_horizontal = abs(self.line_base_pos) + abs(other_line.line_base_pos) 
        dist_okay       = (dist_horizontal > 3.0) and (dist_horizontal < 12.0) 
        
        slope_diff    = self.best_fit[1] - other_line.best_fit[1]
        are_parallel  = abs(slope_diff) < 0.1
                           
        lanes_ok = (radius_similar and dist_okay and are_parallel)
        
        self.detected = lanes_ok
        
        if lanes_ok:
            self.age_predicted = 0
            print('Lanes ok: radius={} dist={} slope={}'.format(diff_curvature, dist_horizontal, slope_diff))
        else:
            print('Lanes not plausible: radius={} dist={} slope={}'.format(diff_curvature, dist_horizontal, slope_diff))
            
        return lanes_ok
    
    def update_params(self, y_eval):
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension
        
        y_val   = np.asarray(self.ally)
        x_val   = np.asarray(self.allx)
        
        new_fit        = np.polyfit(y_val*ym_per_pix, x_val*xm_per_pix, 2)
        new_fitx       = new_fit[0]*(y_val*ym_per_pix)**2 + new_fit[1]*y_val*ym_per_pix + new_fit[2]
        weights        = 1./np.absolute(new_fitx - x_val*xm_per_pix)
        new_fit        = np.polyfit(y_val*ym_per_pix, x_val*xm_per_pix, 2, w=weights)
        
        new_image_fit = np.polyfit(y_val, x_val, 2)
        new_fitx       = new_image_fit[0]*y_val**2 + new_image_fit[1]*y_val + new_image_fit[2]
        weights        = 1./np.absolute(new_fitx - x_val)
        self.image_fit = np.polyfit(y_val, x_val, 2, w=weights)
        
        assert(abs(new_fit[0]) > 0)
        self.radius_of_curvature = ((1 + (2*new_fit[0]*y_eval + new_fit[1])**2)**1.5) \
                                   /np.absolute(2*new_fit[0])
                                  
        if self.detected:
            self.best_fit       = 0.8*self.best_fit + 0.2*new_fit
            self.best_fit_image = 0.8*self.best_fit_image + 0.2*self.image_fit
            self.diffs          = self.current_fit - new_fit
        else:
            self.best_fit        = new_fit
            self.diffs           = np.array([0,0,0], dtype='float') 
            self.best_fit_image  = self.image_fit
            
        self.current_fit = new_fit
            
        x_eval = new_fit[0]*y_eval**2 + new_fit[1]*y_eval + new_fit[2]
        self.line_base_pos = abs(640 - x_eval)*xm_per_pix
        
    def detect_from_scratch(self, warped, lane_idx):
        img_height    = warped.shape[0]
        window_height = int(img_height/10)
        window_width  = 100
        
        # Set minimum number of pixels found to recenter window
        minpix = 50
        
        self.allx = []
        self.ally = []
                           
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero  = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        lane_pix_idx = []
        search_boxes = []
        
        for row in range(0,10):        
            yBot          = img_height - row*window_height
            yTop          = img_height - (row+1)*window_height
            x_left        = int(lane_idx - window_width)
            x_left        = max(x_left, 0)
            x_right       = int(lane_idx + window_width)
            x_right       = min(x_right, warped.shape[1])
            img_window    = warped[yTop:yBot,x_left:x_right]
            
            search_boxes.append((x_left,yBot,x_right,yTop))
    
            good_idx = ((nonzeroy >= yTop) & (nonzeroy < yBot) & (nonzerox >= x_left) & (nonzerox < x_right)).nonzero()[0]
            lane_pix_idx.append(good_idx)
            # Is maximum significant?
            if len(good_idx) > minpix:
                lane_idx = np.int(np.mean(nonzerox[good_idx]))
                
                points_y, points_x = np.nonzero(img_window)
                points_x  += x_left
                points_y  += yTop
                
                self.allx.extend(points_x)
                self.ally.extend(points_y)
                
        # Concatenate the array of indices
        lane_pix_idx = np.concatenate(lane_pix_idx)

        # Extract line pixel positions
        if len(lane_pix_idx) > 0:
            self.allx = nonzerox[lane_pix_idx]
            self.ally = nonzeroy[lane_pix_idx] 
                            
        return search_boxes, len(self.allx) > 100
    
    def detect_from_filter(self, warped):
        
        self.allx = []
        self.ally = []
        
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "warped")
        # It's now much easier to find line pixels!
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        lane_idx = ((nonzerox > (self.best_fit_image[0]*(nonzeroy**2) + self.best_fit_image[1]*nonzeroy + self.best_fit_image[2] - margin)) & (nonzerox < (self.best_fit_image[0]*(nonzeroy**2) + self.best_fit_image[1]*nonzeroy + self.best_fit_image[2] + margin))) 
        
        # Again, extract left and right line pixel positions
        self.allx = nonzerox[lane_idx]
        self.ally = nonzeroy[lane_idx] 
                
        return len(self.allx) > 10
            
