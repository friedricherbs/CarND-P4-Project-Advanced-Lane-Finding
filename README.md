##Advanced Lane Finding Project

This project presents a lane detection and tracking approach implemented in Python. This project was developed and submitted as part of Udacity Self Driving Car Nanodegree.

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: undist.png "Undistorted"
[image2]: undist_scene.png "Undistorted Example"
[image3]: thresholds.png "Thresholding Example"
[image4]: warping.png "Warp Example"
[image5]: hist.png "Histogram example"
[image6]: lane_fit.png "Histogram Lane detection"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Camera Calibration

The camera calibration is implemented in the file [calib.py](https://github.com/friedricherbs/CarND-P4-Project-Advanced-Lane-Finding/blob/master/calib.py). 
In order to correct for distortion artifacts, there are basically two main steps: we use chessboard images to image points and object points, and then use the OpenCV functions cv2.calibrateCamera() and cv2.undistort() to compute the calibration and undistortion.

I start by preparing "objpoints", which will be the (x, y, z) coordinates of the chessboard corners in the 3D world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Distortion correction
The first step in the pipeline in [lane_detection.py](https://github.com/friedricherbs/CarND-P4-Project-Advanced-Lane-Finding/blob/master/lane_detection.py) in line 278 is to use the calibration and distortion coefficients found in the calibration phase to undistort the original image. Therefore again the OpenCV function cv2.undistort() is used. An example image is shown here:

![alt text][image2]

The differences are particularly visible at the image borders due to radial distortion.

####2. Thresholding

As next step, I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 28 and 32 in [threshold.py](https://github.com/friedricherbs/CarND-P4-Project-Advanced-Lane-Finding/blob/master/threshold.py)). Before the thresholding, the image is converted to the more illumination invariant S channel of HLS color space for the color thresholding (line 15 and 16) and to grayscale (line 19) before applying a Sobel filter mask (line 22). Here's an example of my output for this step:  

![alt text][image3]

####3. Perspective Transform

The code for my perspective transform is implemented in the function `perspective_transform()`, which appears in lines 26 through 48 in the file [lane_detection.py](https://github.com/friedricherbs/CarND-P4-Project-Advanced-Lane-Finding/blob/master/lane_detection.py).  The `perspective_transform()` function takes as inputs an image (`image`) and warps the source points to the destination points to undo the projective transformation in the following manner:

```
shape = (image.shape[0], image.shape[1]) 
x_top_right = int(0.6*shape[1])
x_top_left  = int(0.4*shape[1])
y_top       = int(shape[0]/1.5)
src = np.float32(
    [[0,shape[0]],
    [shape[1],shape[0]],
    [x_top_right,y_top],
    [x_top_left,y_top])
    
offset = 100 
dst = np.float32(
    [[offset,shape[0]-offset],
    [shape[1]-offset,shape[0]-offset],
    [shape[1]-offset,offset],
    [offset,offset])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 0, 720        | 100, 620      | 
| 1280, 720     | 1180, 620     |
| 768, 480      | 1180, 100     |
| 512, 480      | 100, 100      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image:

![alt text][image4]

####4. Lane finding and fitting

Basically two approaches for lane detection exist: the first approach implemented in `find_lanes_histogram()` detects the lane from scratch using a histogram approach. The idea for this approach is that after applying calibration, thresholding, and a perspective transform to a road image, the resulting binary image should have clearly outstanding lane markings. Taking a histogram along all the columns of the lower half of the image (see line 71 of [lane_detection.py](https://github.com/friedricherbs/CarND-P4-Project-Advanced-Lane-Finding/blob/master/lane_detection.py) then yields

![alt text][image5]

The peaks of the histogram (line 72 and 73) then serve as a starting point for a sliding window search implemented in the function `detect_from_scratch.py` starting in line 107 in the file [line.py](https://github.com/friedricherbs/CarND-P4-Project-Advanced-Lane-Finding/blob/master/line.py). Basically this function follows the lines up to the top of the frame.

![alt text][image6]

The thresholded input image `warped` is divided into 10 horizontal bands (see green boxes in the figure) and a window around the initial lane position is placed (see line 133). In the case that there are sufficiently many non-zero points inside this window (line 140), the mean position of the points then updates the lane position estimate for this image band (line 141). All non-zero pixels inside the sliding windows are included in the lane pixels called `allx` and `ally`, see line 155 and 156. The pixel coordinates of these lane pixels are refined subsequently in the method `update_params()` in line 72 in  [line.py](https://github.com/friedricherbs/CarND-P4-Project-Advanced-Lane-Finding/blob/master/line.py) by fitting a second order polynomial to the points to find the lane parameters. The fit is performed once for world coordinates (line 82) and once for image coordinates (line 87). A robust fitting scheme with different weights depending on the residual was applied (see weight definition in line 81 and 86).
The second approach uses the filtered values as a starting point for the lane search. 

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
