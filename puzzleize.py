# -*- coding: utf-8 -*-
"""
Created on Sat May 16 15:07:12 2020

@author: Benjamin.Garzon
"""

# fix color changes ??
# mix more

#from __future__ import division
import imutils
import cv2
import os
from random import choice
import copy 
import numpy as np
import glob
from video_puzzle_funcs import create_bck, masking, resize, join_vids, \
merge_vids, create_sequence, apply_fun, time_average, board, cropping, \
    cleaning

FFMPEGOPTIONS='-vb 1000M -qscale 1 -vcodec mpeg4'
       
# run the processing
#files = glob.glob(os.path.join(dir_in, "*.mp4"))
#files = [os.path.basename(f) for f in files]
#dir_in="C:/Users/benjamin.garzon/Google Drive/Colab/neural_style/neural-style-tf/image_input/selected"
dir_in="input"
dir_out="images"
# vertical, horizontal
size = (1080, 1920)
size2 = (1080, 960)
#size = (720, 1280)
#myboard = board(size, (7, 13), (4, 7))
myboard = board(size, (8, 14), (4, 7), maxmicrosteps = 20, offset = (0, 0))

myboard1 = board(size2, (8, 7), (7, 0), maxmicrosteps = 20, offset = (0, 0))
myboard2 = board(size2, (8, 7), (7, 6), maxmicrosteps = 20, offset = (0, 0))
        
colors = \
[ (123,104,238), # medium slate blue
  (128,128,128), # silver
  (154,205,50), # yellow green
  (255,160,122), # light salmon
  (160, 82, 45), # sienna
  (255,255,  0), # yellow
  (128,  0,128), # purple
  (128,128,  0) ] # olive

colors = \
[ (255, 51, 51), #red 2
  (51, 255, 51), # green 1
  (51, 51, 255), # dark blue 2
  (255,153, 51), #orange 1
  (51, 255, 255), # cyan 2
  (255, 51, 255), #purple 1
  (153, 255, 51), # green 2
  ( 51,153,255) # blue 1
 ] 
#  (255, 51, 153), # magenta 2

bck = [ create_bck((int(size[0]/2), int(size[1]/4)), c[0], c[1], c[2]) for c in colors ]

filenames = ["Face2_cut", "Face1_cut", "Face4_cut", "Face3_cut"]
limhs = 2*[(20, 1010),(20, 1010),(20, 1010),(20, 1010)]
#limws = [(80, 1840),(80, 1840),(80, 1840),(80, 1840)]
limws = 2*[(540, 1420),(540, 1420),(540, 1420),(540, 1420)]

#### test different thresholds
#thresholds = [10, 30, 50, 70, 90, 110] # increase to make thinner

#for threshold in thresholds:
#    apply_fun(os.path.join(dir_out, "%s_small.mp4"%(filenames[0])), 
#              os.path.join(dir_out, "%s_mask_%d.mp4"%(filenames[0], threshold)), 
#              masking, skip = 0, background = bck[0], 
#              intensity = .7, 
#              threshold = threshold)

#stophere
#thresholds = [90, 90, 60, 60] # increase to make thinner
thresholds = [50, 50, 50, 50, 30, 30, 30, 30] # increase to make thinner
thresholds = [50, 50, 50, 50, 50, 50, 50, 50] # increase to make thinner
thresholds = [30, 30, 30, 30, 30, 30, 30, 30] # increase to make thinner

#filename = "Face2_cut"
#apply_fun(os.path.join(dir_out, "%s_small.mp4"%(filename)), 
#              os.path.join(dir_out, "%s_test.mp4"%(filename)), 
#              masking, skip = 0, background = bck.pop(), 
#              intensity = 1.2, 
#              threshold = thresholds.pop())
if False:
    for index, filename in enumerate(filenames): 
        # crop
        apply_fun(os.path.join(dir_in, "%s.mp4"%(filename)), 
                  os.path.join(dir_out, "%s_crop.mp4"%(filename)), 
                  cropping, skip = 116, limh = limhs.pop(), limw = limws.pop(), 
                  flip = False)
    
        # crop
        apply_fun(os.path.join(dir_in, "%s.mp4"%(filename)), 
                  os.path.join(dir_out, "%s_crop_flipped.mp4"%(filename)), 
                  cropping, skip = 116, limh = limhs.pop(), limw = limws.pop(), 
                  flip = True)
    
        # resize
        apply_fun(os.path.join(dir_out, "%s_crop.mp4"%(filename)), 
                  os.path.join(dir_out, "%s_small.mp4"%(filename)), 
                  resize, skip = 0, height = int(size[0]/2), width = int(size[1]/4) )    
    
        apply_fun(os.path.join(dir_out, "%s_crop_flipped.mp4"%(filename)), 
                  os.path.join(dir_out, "%s_small_flipped.mp4"%(filename)), 
                  resize, skip = 0, height = int(size[0]/2), width = int(size[1]/4) )    


# change duration
#    change_duration(os.path.join(dir_out, "%s_small.mp4"%(filename)), 
#              os.path.join(dir_out, "%s.mp4"%(filename)), 
#              factor = 1.6)
if True:
    for index, filename in enumerate(filenames): 
        # mask
        apply_fun(os.path.join(dir_out, "%s_small.mp4"%(filename)), 
                      os.path.join(dir_out, "%s_mask.mp4"%(filename)), 
                      masking, skip = 0, background = bck.pop(), 
                      intensity = 1.2, 
                      threshold = thresholds.pop())
    
        apply_fun(os.path.join(dir_out, "%s_small_flipped.mp4"%(filename)), 
                      os.path.join(dir_out, "%s_mask_flipped.mp4"%(filename)), 
                      masking, skip = 0, background = bck.pop(), 
                      intensity = 1.2, 
                      threshold = thresholds.pop())
       
    join_vids(os.path.join(dir_out, "%s_mask.mp4"%(filenames[0])), 
              os.path.join(dir_out, "%s_mask.mp4"%(filenames[1])), 
              os.path.join(dir_out, "upper1.mp4"))
    
    join_vids(os.path.join(dir_out, "%s_mask_flipped.mp4"%(filenames[1])), 
              os.path.join(dir_out, "%s_mask_flipped.mp4"%(filenames[0])), 
              os.path.join(dir_out, "upper2.mp4"))

    join_vids(os.path.join(dir_out, "%s_mask.mp4"%(filenames[2])), 
              os.path.join(dir_out, "%s_mask.mp4"%(filenames[3])), 
              os.path.join(dir_out, "lower1.mp4"))
    
    join_vids(os.path.join(dir_out, "%s_mask_flipped.mp4"%(filenames[3])), 
              os.path.join(dir_out, "%s_mask_flipped.mp4"%(filenames[2])), 
              os.path.join(dir_out, "lower2.mp4"))
        
    join_vids(os.path.join(dir_out, "upper1.mp4"),
              os.path.join(dir_out, "lower1.mp4"), 
              os.path.join(dir_out, "left.mp4"), horiz = False)
    
    join_vids(os.path.join(dir_out, "upper2.mp4"),
              os.path.join(dir_out, "lower2.mp4"), 
              os.path.join(dir_out, "right.mp4"), horiz = False)


time_average(os.path.join(dir_out, "right.mp4"), 
              os.path.join(dir_out, "right_average.mp4"))

time_average(os.path.join(dir_out, "left.mp4"), 
              os.path.join(dir_out, "left_average.mp4"))

#        apply_fun(os.path.join(dir_out, "complete_sequence2.mp4"), 
#                      os.path.join(dir_out, "complete_sequence2_clean.mp4"), 
#                      cleaning, skip = 0, threshold = 200)
 
# puzzleize
mysequence = create_sequence(myboard, duration = 240.0)
mysequence.reverse()

mysequence1 = create_sequence(myboard1, duration = 158.0)
mysequence2 = create_sequence(myboard2, duration = 158.0)
mysequence1.backandforth()
mysequence2.backandforth()


apply_fun(os.path.join(dir_out, "left_average.mp4"), 
      os.path.join(dir_out, "left_sequence.mp4"), 
      mysequence1.apply_to_vid, reverse = False)

apply_fun(os.path.join(dir_out, "right_average.mp4"), 
      os.path.join(dir_out, "right_sequence.mp4"), 
      mysequence2.apply_to_vid, reverse = False)

join_vids(os.path.join(dir_out, "left_sequence.mp4"),
          os.path.join(dir_out, "right_sequence.mp4"), 
          os.path.join(dir_out, "complete_sequence2.mp4"), padding = True)

apply_fun(os.path.join(dir_out, "complete_sequence2.mp4"), 
    os.path.join(dir_out, "complete_sequence2_clean.mp4"), 
    cleaning, skip = 0, threshold = 200)

f = os.path.join(dir_out, "complete_sequence2_audio.mp4")
if os.path.exists(f):
    os.remove(f)

os.system("ffmpeg -i %s\\complete_sequence2_clean.mp4 -i \"C:\\Users\\benjamin.garzon\\Google Drive\\Music\\EP2014\\Wav\\Different.wav\" -map 0:v -map 1:a %s -shortest %s\\complete_sequence2_audio.mp4"%(dir_out, FFMPEGOPTIONS, dir_out))




if False:
    join_vids(os.path.join(dir_out, "left_average.mp4"),
              os.path.join(dir_out, "right_average.mp4"), 
              os.path.join(dir_out, "complete.mp4"), horiz = True)

    # puzzleize
    apply_fun(os.path.join(dir_out, "complete.mp4"), 
              os.path.join(dir_out, "complete_sequence.mp4"), 
              mysequence.apply_to_vid)

    # build replicated version
    apply_fun(os.path.join(dir_out, "complete.mp4"), 
              os.path.join(dir_out, "complete_small.mp4"), 
              resize, skip = 0, height = int(size[0]/4), width = int(size[1]/4) )    

    join_vids(os.path.join(dir_out, "complete_small.mp4"),
              os.path.join(dir_out, "complete_small.mp4"), 
              os.path.join(dir_out, "complete_large.mp4"), horiz = False)

    join_vids(os.path.join(dir_out, "complete_large.mp4"),
              os.path.join(dir_out, "complete_large.mp4"), 
              os.path.join(dir_out, "complete_double.mp4"), horiz = True)

    # build replicated version
    apply_fun(os.path.join(dir_out, "complete_double.mp4"), 
              os.path.join(dir_out, "complete_double_small.mp4"), 
              resize, skip = 0, height = int(size[0]/4), width = int(size[1]/4) )    

    join_vids(os.path.join(dir_out, "complete_double_small.mp4"),
              os.path.join(dir_out, "complete_double_small.mp4"), 
              os.path.join(dir_out, "complete_double_large.mp4"), horiz = False)

    join_vids(os.path.join(dir_out, "complete_double_large.mp4"),
              os.path.join(dir_out, "complete_double_large.mp4"), 
              os.path.join(dir_out, "complete_quadruple.mp4"), horiz = True)

    # merge videos
    merge_vids(os.path.join(dir_out, "complete_sequence2.mp4"), 
              os.path.join(dir_out, "complete_quadruple.mp4"), 
              os.path.join(dir_out, "final.mp4"),
              (0.0, 170.0), (170.0, 400.0))

    apply_fun(os.path.join(dir_out, "complete_sequence.mp4"), 
        os.path.join(dir_out, "complete_sequence_clean.mp4"), 
        cleaning, skip = 0, threshold = 250)

    os.remove(os.path.join(dir_out, "complete_audio.mp4"))
    os.remove(os.path.join(dir_out, "complete_sequence_audio.mp4"))
    os.system("ffmpeg -i %s\\complete.mp4 -i \"C:\\Users\\benjamin.garzon\\Google Drive\\Music\\EP2014\\Wav\\Different.wav\" -map 0:v -map 1:a %s -shortest %s\\complete_audio.mp4"%(dir_out, FFMPEGOPTIONS, dir_out))
    os.system("ffmpeg -i %s\\complete_sequence_clean.mp4 -i \"C:\\Users\\benjamin.garzon\\Google Drive\\Music\\EP2014\\Wav\\Different.wav\" -map 0:v -map 1:a %s -shortest %s\\complete_sequence_audio.mp4"%(dir_out, FFMPEGOPTIONS, dir_out))
