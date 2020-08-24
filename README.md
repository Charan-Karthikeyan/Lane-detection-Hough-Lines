[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Lane-detection-Hough-Lines
This project was done to fulfill the requirements of the CARND nannodree from Udacity.

## Overview
The Objective of the repo is to showcase how to detect the lane lines on a road using Hough lines and Perspective Transformation to extract the required information from them. This project is a sub-module of the Self-Driving cars Visual System. The project was programmed using Python3 libraries.

## Purpose
The Lane Lines on a road denote the designated path on which the vehicle has to traverse on as the actor module of the Path-Planning module of the Autonomous Vehicle. The Planning module of the self driving car is dependant on various sensors such as  1.)Visual Perception Sensors(Cameras)  2.)LIDAR  3.) RADAR  4.)SONAR or Ultrasonics  . This repo is part of the Visual Perception Sensor input to the PLanning moudle. This is a time critical process and can be used to steer the vehicle using Real-Time data.
The examples of how the lane lines are shown in the below image.

## Hough Lines
Hough Transform is a mehodology that is used to detect any shape, if that particular shape can be depicted in a mathematical format.
In this project we will be using the Hough lines to detect the lane lines and device a mathematical way to draw and connect the points.

## Project Requirements
1.) Python3  2.) OpenCV  3.)Numpy 4.) Matplot Libraries  5.)Moviepy  6.) IPython  7.) ffmpeg dependencies

## Pipeline

The following steps are done in a step by step way to get the desired result:
1.) Extract each frame from the video file.
2.) Convert the frame to a grayscale image 
3.) The converted image is then passed through a gaussian filter and passed through kernel then passed through the blurring filter.
4.) The filtered image is then sent through the canny filter and then thresholded with the high and low thershold.
5.) The vertices which is required is found using the upper, lower and the apex of the vertices values. The ROI is selected using the value of the vertices on the frame.
6.) The ROI is then passed through Hough filter which draws the lane lines. The lane lines are drawn using the rho, theta, threshold, min_line_lenght and the max_line_gap parameters are defined and sent in this function.
7.) The lines are then drawn on the original image using the weighted function.

## Command to run code
python3 Lane_Lines.py

## References
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html

## Licence
The Repository is Licensed under the MIT License.
```
MIT License

Copyright (c) 2019 Charan Karthikeyan Parthasarathy Vasanthi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Known Issue or Bugs
One potential shortcoming would be what would happen when the lines contnuous flickering when the video encounters curved roads 

Another shortcoming could be the hough transform is slow and can be visibily seen 

## Possible Improvements 
Decrease the size of the ROI and to make the procesing much faster and smoother.

