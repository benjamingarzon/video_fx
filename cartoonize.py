# -*- coding: utf-8 -*-
"""
Created on Sat May  9 16:48:25 2020

@author: Benjamin.Garzon
"""
import os
import cv2
import imutils
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer
import numpy as np
import glob
#20, 66, 112

#dir_in="C:/Users/benjamin.garzon/Google Drive/Colab/neural_style/neural-style-tf/image_input/selected/"
dir_in="C:/Users/benjamin.garzon/Desktop/selected"
dir_out="images"


def cartoonize(file_in, smoothing = 3):
    print(file_in)
    path_in=os.path.join(dir_in, file_in)
    path_out=os.path.join(dir_out, file_in)
    if os.path.exists(path_out):
        os.remove(path_out)

    video_clip = VideoFileClip(path_in, audio=False)

    
    i = 0
    for img in video_clip.iter_frames():
        print(i)
        i += 1
#        if i == 100:
#            break
    
        # 1) Edges
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        
        # 2) Color
        color = cv2.bilateralFilter(img, 9, 200, 200)
        
        # 3) Cartoon
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        output = cv2.GaussianBlur(cartoon, (smoothing, smoothing), 0)
 #       output = cartoon
        cv2.imwrite("images/im%04d.png"%(i), output)

    fps = int(video_clip.fps)    
    cv2.destroyAllWindows()
    os.system("ffmpeg -r %d -i images/im%%04d.png -vb 40M -vcodec mpeg4 -r %d %s"%(fps, fps, path_out))
    files = glob.glob(os.path.join(dir_out, "*.png"))
    for f in files:
        os.remove(f)
        
# run the processing
files = glob.glob(os.path.join(dir_in, "*.mp4"))
files = [os.path.basename(f) for f in files]
#cartoonize("chorus2a.mp4")
for f in files:
    cartoonize(f)
