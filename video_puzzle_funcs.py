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

FFMPEGOPTIONS='-vb 1000M -qscale 1 -vcodec mpeg4'


class piece:
    def __init__(self, pos):
        self.pos = pos # position in the original image
        self.visits = 0
    
    def get_freqs(self):
        return(int(10000/(1 + 5*self.visits)))
    
    def add_visit(self):
        self.visits += 1
        
class board:

    def __init__(self, size, n, init, maxmicrosteps, offset = (0, 0)):
        self.size = size # board size
        self.n = n # number of rows, columns
        self.missing = init # position of missing piece
        self.prevmissing = None
        self.sizex = int(size[0]/n[0])
        self.sizey = int(size[1]/n[1])
        self.stepn = 0
        self.step = None
        self.microstep = 0        
        self.maxmicrosteps = maxmicrosteps
        self.offset = offset

        self.piecelist = []        
        for ix in range(n[0]):
            ylist = []
            for iy in range(n[1]):
                pos = (self.sizex*ix, self.sizey*iy)
                ylist.append(piece(pos))                 
            self.piecelist.append(ylist)

    def print(self):
        n = self.n
        for ix in range(n[0]):
            for iy in range(n[1]):        
                pos = self.piecelist[ix][iy].pos
                print("%d, %d -> (%d, %d) %d visits "%(ix, iy, pos[0], pos[1], self.piecelist[ix][iy].visits))
               
    def score(self):
        n = self.n
        
        quad = np.zeros(4)
        for ix in range(n[0]):
            for iy in range(n[1]):
                pos = self.piecelist[ix][iy].pos
                size = self.size
                if (ix < n[0]/2 and iy < n[1]/2) and not (pos[0] < size[0]/2 and pos[1] < size[1]/2):
                   quad[0] += 1
                if (ix < n[0]/2 and iy >= n[1]/2) and not (pos[0] < size[0]/2 and pos[1] >= size[1]/2):
                   quad[1] += 1
                if (ix >= n[0]/2 and iy < n[1]/2) and not (pos[0] >= size[0]/2 and pos[1] < size[1]/2):
                   quad[2] += 1
                if (ix >= n[0]/2 and iy >= n[1]/2) and not (pos[0] >= size[0]/2 and pos[1] >= size[1]/2):
                   quad[3] += 1
                #print(quad)   
        return(np.min(quad))
        
    def move(self):
        
        self.stepn += 1
        m = self.missing
        aux = self.piecelist[m[0]][m[1]]

        stepchoices = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        if m[0] == 0:
            stepchoices.remove((-1, 0))
        if m[1] == 0:
            stepchoices.remove((0, -1))
        if m[0] == self.n[0]-1:
            stepchoices.remove((1, 0))
        if m[1] == self.n[1]-1:
            stepchoices.remove((0, 1))
       
        # don't allow going back
        if self.step == (0, 1):
            stepchoices.remove((0, -1))
        if self.step == (0, -1):
            stepchoices.remove((0, 1))
        if self.step == (1, 0):             
            stepchoices.remove((-1, 0))
        if self.step == (-1, 0):
            stepchoices.remove((1, 0))
  
        freqs = [ self.piecelist[m[0] + step[0]][m[1] + step[1]].get_freqs() for step in stepchoices ]
        choicelist = []
        for i, f in enumerate(freqs):
            choicelist.extend(f*[stepchoices[i]])
        #print(freqs)
        step = choice(choicelist)

        new = (m[0] + step[0], m[1] + step[1])
        aux = self.piecelist[new[0]][new[1]]
        self.piecelist[new[0]][new[1]] = self.piecelist[m[0]][m[1]] 
        self.piecelist[m[0]][m[1]] = aux
        self.prevmissing = self.missing
        self.missing = new
        self.step = step
        
        # add a visit to former position
        self.piecelist[m[0]][m[1]].add_visit()
        
    def apply_to_img(self, img, show = True, grid = True, slide = True):
        n = self.n
        sx = self.sizex
        sy = self.sizey
        offset = self.offset
        newimg = img.copy()*0
        for ix in range(n[0]):
            for iy in range(n[1]):
                pos = self.piecelist[ix][iy].pos
                newpos = (offset[0] + sx*ix, offset[1] + sy*iy)
                newimg[newpos[0]:(newpos[0] + sx), \
                       newpos[1]:(newpos[1] + sy), :] = \
                img[(offset[0] + pos[0]):(offset[0] + pos[0] + sx ), \
                    (offset[1] + pos[1]):(offset[1] + pos[1] + sy ), :]
                    
        # make a grid        
        if grid: 
            for ix in range(n[0]):
                for iy in range(n[1]):
                    newpos = (sx*ix, sy*iy)
                    newimg[newpos[0], :,  :] = 0
                    newimg[:, newpos[1],  :] = 0

        # empty the missing ones
        newpos = (sx*self.missing[0], sy*self.missing[1])
        newimg[newpos[0]:(newpos[0]+sx), \
               newpos[1]:(newpos[1]+sy), :] = 0            

        
        # slide piece
        
        if self.step != None and slide:
            newpos = (sx*self.prevmissing[0], sy*self.prevmissing[1])
            newimg[newpos[0]:(newpos[0]+sx), \
                   newpos[1]:(newpos[1]+sy), :] = 0            

            k = self.microstep/self.maxmicrosteps 
            pos = self.piecelist[self.prevmissing[0]][self.prevmissing[1]].pos

            newpos = (int(sx*(self.prevmissing[0] + k*self.step[0])), 
                     int(sy*(self.prevmissing[1] + k*self.step[1])))

            newimg[newpos[0]:(newpos[0]+sx), \
                newpos[1]:(newpos[1]+sy), :] = \
                   img[pos[0]:(pos[0]+sx), \
                       pos[1]:(pos[1]+sy), :]

            if grid:             
                newimg[newpos[0]:(newpos[0]+sx), \
                    newpos[1], :] = 0
    
                newimg[newpos[0], \
                    newpos[1]:(newpos[1]+sy), :] = 0
            
            if self.microstep < self.maxmicrosteps:
                self.microstep += 1
            
        if show:
            cv2.imshow("New", newimg)
            # cv2.imshow("Original", img)
            
        return(newimg)
        
    def copy(self):
        return copy.deepcopy(self)

class sequence:
    def __init__(self, b, step_pars):
        self.boards = []
        self.start = step_pars[0]
        self.end  = step_pars[1]
        self.step_size = step_pars[2]
        steps = int((self.end - self.start)/self.step_size)
        self.steps = steps

        for i in range(steps):
            self.boards.append(b.copy())
            b.move()
    
    def print_missing(self, reverse = False):
        boards = self.boards[::-1] if reverse else self.boards
        for i, x in enumerate(boards):
            print(i, x.missing, x.prevmissing, x.step)
    
    def play(self, img, wait, reverse = False):
        boards = self.boards[::-1] if reverse else self.boards
        for i, b in enumerate(boards):
            b.apply_to_img(img)
            cv2.waitKey(wait)

    def backandforth(self):
        boards = [ x.copy() for x in self.boards[::-1] ] + \
            [ x.copy() for x in self.boards ]
        self.boards = boards
        
    def reverse(self):
        boards = [ x.copy() for x in self.boards[::-1]]
        self.boards = boards
        
    def apply_to_vid(self, img, kwargs):
        mytime = kwargs['time'] 

#        reverse = kwargs['reverse']
#        if reverse: 
#            self.reverse()
#        boards = self.boards[::-1] if reverse else self.boards
        n = np.max((
            np.min(
            (int( (mytime - self.start) / self.step_size ), self.steps - 1)), 
            0 ) 
            )
        
        
        newimg = self.boards[n].apply_to_img(img, show = False, 
                                        slide = True if n > 0 else False)
        
        return(newimg)


# functions

def add_overlay(frame, overlay, intensity):
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGRA)
    frame = verify_alpha_channel(frame)
    cv2.addWeighted(overlay, 
                    intensity, 
                    frame, 1.0, 0, 
                    frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame

def apply_color_overlay(frame, 
            intensity=0.2, 
            blue = 0,
            green = 0,
            red = 0):
    frame = verify_alpha_channel(frame)
    frame_h, frame_w, frame_c = frame.shape
    color_bgra = (blue, green, red, 1)
    overlay = np.full((frame_h, frame_w, 4), color_bgra, dtype='uint8')
    cv2.addWeighted(overlay, intensity, frame, 1.0, 0, frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame

def create_bck(frame_size, 
            red = 0,
            green = 0,
            blue = 0):
    color_bgra = (blue, green, red, 1)
    bck = np.full((frame_size[0], frame_size[1], 4), color_bgra, dtype='uint8')
    return bck

def apply_sepia(frame, intensity=0.5):
    blue = 20
    green = 66 
    red = 112
    frame = apply_color_overlay(frame, 
            intensity=intensity, 
            blue=blue, green=green, red=red)
    return frame

def join_vids(path_in1, path_in2, path_out, horiz = True, padding = False):
    
    dir_out = os.path.dirname(path_out)
    
    print(path_out)
    if os.path.exists(path_out):
        os.remove(path_out)

    video_clip1 = cv2.VideoCapture(path_in1)
    video_clip2 = cv2.VideoCapture(path_in2)

    fps = int(video_clip1.get(cv2.CAP_PROP_FPS)) 
            
    i = 0
    while(video_clip1.isOpened()):
        #print(i)
        ret1, img1 = video_clip1.read()
        ret2, img2 = video_clip2.read()
        if (not ret1) | (not ret2):
            break

        if padding:
            # to avoid a thicker line in the middle
            img2[:, 0:-1] = img2[:, 1:]
            
        output = cv2.hconcat([img1, img2]) if horiz else cv2.vconcat([img1, img2]) 
        cv2.imwrite(os.path.join(dir_out, "im%04d.png"%(i)), output)
        i += 1
        
#    cv2.destroyAllWindows()
    os.system("ffmpeg -r %d -i images/im%%04d.png %s -r %d %s"%(fps, FFMPEGOPTIONS, fps, path_out))
    files = glob.glob(os.path.join(dir_out, "*.png"))
    for f in files:
        os.remove(f)

def average_frames(images):
    avg_img = np.mean(images, axis=0).astype(int)
    return(avg_img)

def time_average(path_in, path_out, NFRAMES = 5):
    
    NMAX = np.ceil(NFRAMES/2).astype(int)
    NF = NMAX
    dir_out = os.path.dirname(path_out)
    
    print(path_out)
    if os.path.exists(path_out):
        os.remove(path_out)

    video_clip = cv2.VideoCapture(path_in)

    fps = int(video_clip.get(cv2.CAP_PROP_FPS)) 
    
    i = 0
    frames = []
    while(video_clip.isOpened()):
        #print(i)
        while(len(frames) < NF):
            ret, img = video_clip.read()
            if (not ret):
                break
            frames.append(img)
            
        if (not ret):
            if len(frames < NF):
                break
        
        output = average_frames(frames)

        if NF < NFRAMES:
            NF = NF + 1
        else:
            frames.pop(0)
        
        cv2.imwrite(os.path.join(dir_out, "im%04d.png"%(i)), output)
        i += 1
        
#    cv2.destroyAllWindows()
    os.system("ffmpeg -r %d -i images/im%%04d.png %s -r %d %s"%(fps, FFMPEGOPTIONS, fps, path_out))
    files = glob.glob(os.path.join(dir_out, "*.png"))
    for f in files:
        os.remove(f)


def merge_vids(path_in1, path_in2, path_out, lim1, lim2):
    
    dir_out = os.path.dirname(path_out)
    
    print(path_out)
    if os.path.exists(path_out):
        os.remove(path_out)

    video_clip1 = cv2.VideoCapture(path_in1)
    video_clip2 = cv2.VideoCapture(path_in2)

    fps = int(video_clip1.get(cv2.CAP_PROP_FPS)) # assume same for both
    print(fps)
    if isinstance(lim1[0], float): # given in seconds
        lim1 = int(lim1[0]*fps), int(lim1[1]*fps)
    if isinstance(lim2[0], float):
        lim2 = int(lim2[0]*fps), int(lim2[1]*fps)
    
    i = 0
    j = 0
    while(video_clip1.isOpened()):
        ret, img = video_clip1.read()
        print(i, j)
        if (not ret):
            break
        
        if i > lim1[0]:
            cv2.imwrite(os.path.join(dir_out, "im%04d.png"%(j)), img)
            j += 1
        if i == lim1[1]:
            break
        i += 1

    i = 0
    while(video_clip2.isOpened()):
        ret, img = video_clip2.read()
        print(i, j)
        if (not ret):
            break
        if i > lim2[0]:
            cv2.imwrite(os.path.join(dir_out, "im%04d.png"%(j)), img)
            j += 1
        if i == lim2[1]:
            break
        i += 1
        
    os.system("ffmpeg -r %d -i images/im%%04d.png -vb 1000M -vcodec mpeg4 -r %d %s"%(fps, fps, path_out))
    files = glob.glob(os.path.join(dir_out, "*.png"))
    for f in files:
        os.remove(f)


def apply_invert(frame):
    return cv2.bitwise_not(frame)

def verify_alpha_channel(frame):
    try:
        frame.shape[3] # 4th position
    except IndexError:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    return frame

def resize(img, kwargs):
    height = kwargs['height']
    width = kwargs['width']
    img = imutils.resize(img, width = width, height=height)
    return(img)

def cropping(img, kwargs):
    limh = kwargs['limh']
    limw = kwargs['limw']
    flip = kwargs['flip']
    out = img[limh[0]:limh[1], limw[0]:limw[1], :]
    if flip:
        out = out [:, ::-1, :]
    return(out)

def get_edges(img):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    newimg = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    newimg = cv2.morphologyEx(newimg, cv2.MORPH_OPEN, kernel)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    
    # take morphological gradient
    gradient_image = cv2.morphologyEx(newimg, cv2.MORPH_GRADIENT, kernel)
    
    # split the gradient image into channels
    image_channels = np.split(np.asarray(gradient_image), 3, axis=2)
    
    channel_height, channel_width, _ = image_channels[0].shape

    # apply Otsu threshold to each channel
    for i in range(0, 3):
        _, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))
    
    # merge the channels
    image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)

    edges = np.min(image_channels, axis = 2)
    #cv2.imshow("New", edges)
    #cv2.waitKey(1000)
    return(edges)

def cleaning(img, kwargs):
    threshold = kwargs['threshold']
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    img[mask == 255] = 255
    return(img)


def masking(img, kwargs):
    background = kwargs['background']
    intensity = kwargs['intensity'] # of background
    threshold = kwargs['threshold']

    edges = get_edges(img)
    img = verify_alpha_channel(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    #blur = cv2.GaussianBlur(gray,(21,21),0)
    #_, mask = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)#+cv2.THRESH_OTSU)
    
    #edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #cv2.THRESH_BINARY, 3, 2)
    #mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    newimg = add_overlay(background, img, intensity = intensity)
#    cv2.imshow('f', mask)
#    cv2.waitKey(1000)
#    cv2.destroyAllWindows
    mask = cv2.bitwise_and(mask, edges)

    newimg[mask == 255] = 255 
    newimg = cv2.bilateralFilter(newimg, 2, 50, 50) #9, 200, 200)
    #dst_gray, newimg = cv2.pencilSketch(newimg, 
    #                                        sigma_s=5, 
    #                                        sigma_r=10, 
    #                                        shade_factor=0.04)
    #newimg = create_pointillism_art(newimg, 10)

    #newimg = cv2.stylization(newimg, sigma_s = 2, sigma_r = 0.1)
    
    return(newimg)

def myFun(**kwargs):  
    for key, value in kwargs.items(): 
        print ("%s == %s" %(key, value))

def apply_fun(path_in, path_out, fun, skip = 0, **kwargs):
    print(path_in)
    print(path_out)
    dir_out = os.path.dirname(path_out)
    
    if os.path.exists(path_out):
        os.remove(path_out)
    video_clip = cv2.VideoCapture(path_in)
    fps = int(video_clip.get(cv2.CAP_PROP_FPS)) 
    if isinstance(skip, float):
        skip = int(skip*fps)
    i = 0

    while(video_clip.isOpened()):
        ret, img = video_clip.read()
        if not ret:
            break
#        print(i)
#        if i == skip + 500:
#            break
        kwargs['time'] = i/fps
        output = fun(img, kwargs)
        if i >= skip:
            cv2.imwrite(os.path.join(dir_out, "im%04d.png"%(i - skip)), output)
        i += 1

    video_clip.release()
    cv2.destroyAllWindows()
    os.system("ffmpeg -r %d -i images/im%%04d.png %s -r %d %s"%(fps, FFMPEGOPTIONS, fps, path_out))
    files = glob.glob(os.path.join(dir_out, "*.png"))
    for f in files:
        os.remove(f)

def change_duration(path_in, path_out, factor = 1.):
    print(path_in)
    print(path_out)
    if os.path.exists(path_out):
        os.remove(path_out)
    os.system("ffmpeg -i %s -filter:v \"setpts=%f*PTS\" %s"%(path_in, factor, path_out))

def create_sequence(myboard, duration):
    maxscore = 0
    for j in range(1000):
        test_sequence = sequence(myboard.copy(), step_pars = (3.1, duration, 60/116))
        if test_sequence.boards[-1].score() > maxscore:
            mysequence = test_sequence
            maxscore = test_sequence.boards[-1].score()
            print(maxscore)
    return(mysequence)
       
