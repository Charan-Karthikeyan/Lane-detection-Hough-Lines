"""
@file : Lane_Line.py
@author : Charan karthikeyan P V
@License : MIT License (c) 2020
@date : 08/15/2020
@brief : This file is to use the video provided and detect the lane lines on them.
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import math
from operator import itemgetter

"""
@brief : Function to connvert the images to grayscale
@param : img -> The image to be converted to grayscale
@return : The image in grayscale
"""
def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray

"""
@brief : Function to find the canny edges in thee image
@param : img -> The image to detect the canny edges
         low_threshoold -> The lower boundary of the threshold for the canny edges
         high_threshold -> The upper boundary of the threshold for the canny edges
@return : Image with the canny edges.
"""
def canny(img, low_threshold, high_threshold):
    # Applies the Canny transform
    cann = cv2.Canny(img, low_threshold, high_threshold)
    return cann

"""
@brief : Function to apply gaussian filter to the image 
@param : img -> The input image to apply the filter
         kernel_size -> The kernel size of the gaussian blur operator
@return : The image with the gaussian blur applied on them.
"""
def gaussian_blur(img, kernel_size):
    # Applies a Gaussian Noise kernel for better acccuracy 
    gauss = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return gauss

"""
@brief : Function to apply perspective transform and get the ROI for the lanes
@param : img -> The input image 
         vertices -> The points where the lane lines are located in the image 
@return : Masked image with the ROI containing the lane markers
"""
def region_of_interest(img, vertices):

#defining a blank mask to start with
    mask = np.zeros_like(img) 

#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2] # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

#filling pixels inside the polygon defined by "vertices" with the fill color 
    cv2.fillPoly(mask, vertices, ignore_mask_color)

#returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

"""
@brief : Function to draw lines for the lane lines
@param : img -> input image with the lane lines after ROI extraction
         lines -> The input of all the points to connect together
         color -> The color of the area between the lines
         thickness -> The thickness of the line to be drawn
@return : None
"""
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):

    

    l_slope=[]
    l_lines=[]
    r_slope=[]
    r_lines=[]

    for line in lines:
        for x1, y1, x2, y2 in line:
            m = ((y2-y1)/(x2-x1)) #slope
            if (m < -0.2) and (m > -0.8):
                l_slope.append(m)
                l_lines.append([x1,y1])
                l_lines.append([x2,y2])

            elif (m > 0.2 ) and (m < 0.8):
                r_slope.append(m)
                r_lines.append([x1,y1])
                r_lines.append([x2,y2])

#find min y values
    ymin_l=min(l_lines, key=itemgetter(1))[1]
    ymin_r=min(r_lines,key=itemgetter(1))[1]
    ymin=min(ymin_l,ymin_r)

#find max y values
    ymax_l=max(l_lines, key=itemgetter(1))[1]
    ymax_r=max(r_lines, key=itemgetter(1))[1]

    ymax=max(ymax_l,ymax_r)


#average left and right slopes
    r_slope = np.mean(r_slope)
    l_slope = np.mean(l_slope)
    #find min x values
    xmin_l=min(l_lines, key=itemgetter(0))[0]
    xmin_r=min(r_lines, key=itemgetter(0))[0]

#find y intercepts
    l_yint=ymax_l-(xmin_l*l_slope)
    r_yint=ymin_r-(xmin_r*r_slope)

    xmax_l=int((ymin_l-l_yint)/l_slope)
    xmax_r=int((ymax_r-r_yint)/r_slope)

    if ymax>ymax_l:
        xmin_l2=int(((ymax-ymax_l)/l_slope)+xmin_l)
        xmin_l=xmin_l2
    elif ymax>ymax_r:
        xmax_r2=int(((ymax-ymax_r)/r_slope)+xmax_r)
        xmax_r=xmax_r2

    if ymin>ymin_l:
        xmax_l2=int(((ymin-ymin_l)/l_slope)+xmax_l)
        xmax_l=xmax_l2
    elif ymin>ymin_r:
        xmin_r2=int(((ymin-ymin_r)/r_slope)+xmin_r)
        xminx_r=xmin_r2


    cv2.line(img, (xmin_l, ymax), (xmax_l, ymin), color, thickness)
    cv2.line(img, (xmin_r, ymin), (xmax_r, ymax), color, thickness)

"""
@brief : Function to draw the hough lines to connect the lane markers
@param : img -> The input image
         rho, theta, threshold -> Parameters of the hough lines
         min_line_len, max_line_gap -> The filter to connect the irregular lines
"""
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img
"""
@brief : Funtion to a add weights to the image
@param : img -> The masked image with the lane markers
         initial_img -> The original image without the mask
         α,β,λ -> The parameters for the weighted image function(default values)
@return : The weighted image.
"""
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    weight = cv2.addWeighted(initial_img, α, img, β, λ)
    return weight


"""
@brief : Function to take in the image and draw the hough lines
@param : Image -> The image to be processed
@return : The final image with the lane lines
"""
def process_image(image):
    np.pi = 3
    img=image
    kernel_size = 5
    rho = 1
    theta = np.pi/180 
    #theta = 0.017
    threshold = 20
    min_line_length = 10
    max_line_gap = 20

    ysize=img.shape[0]
    xsize=img.shape[1]
    #region mask initialize
    left_bottom=[100,ysize]
    right_bottom=[xsize,ysize]
    left_top=[450,320]
    right_top=[550,320]

    vertices=np.array([[left_bottom,left_top,right_top,right_bottom]],dtype=np.int32)

    gray=grayscale(img)
    gauss=gaussian_blur(gray,kernel_size)
    cann=canny(gauss,100,150)
    roi=region_of_interest(cann,vertices)
    hough=hough_lines(roi,rho,theta,threshold,min_line_length,max_line_gap)
    weight=weighted_img(hough,img)
    
    result=weight
    return result

# The code to apply the process on the white broken lane markers video
white_output = 'Output_Video/white.mp4'
clip1 = VideoFileClip("Video/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
# The code to apply the process on the Yellow solid lane markers video
yellow_output = 'Output_Video/yellow.mp4'
clip2 = VideoFileClip('Video/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)