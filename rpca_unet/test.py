# -*- coding: utf-8 -*-

import sys
import torch
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from collections import deque
import numpy as np
import itertools
import re
from classes.Dataset import Converter
from torch.autograd import Variable

import datetime
from matplotlib import animation


sys.path.append('../')
from network.RPCAUNet import RPCAUNet


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def str2int(v_str):
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]


def sort_humanly(v_list):
    return sorted(v_list, key=str2int)


def to_var(X, CalInGPU):
    if CalInGPU and torch.cuda.is_available():
        X = X.cuda()
    return Variable(X)


def generate_single_sequence(dicName="./Data/rawData/311"):
    """Settings"""
    # input folder
    dicList = dicName.split('/')
    testName = dicList[-1]
    # output folder
    filePath = './Results/with_detr_test_10frames/'+ testName + '_CLSTM'

    # estimate whether the output folder exists
    if not os.path.exists(filePath):
        os.makedirs(filePath)
    """========================================================================="""
    # Model file
    mfile = './Results/XCA_RPCA-UNet_Model_al0.40_Tr12000_epoch50_lr1.00e-04_5frames.pkl'

    CalInGPU = True

    """Network Settings: Remember to change the parameters when you change model!"""
    params_net = {'layers': 4,
                  'kernel': [(5, 1)] * 2 + [(3, 1)] * 2,
                  'coef_L': 0.4,
                  'coef_S': 1.8,
                  'CalInGPU': CalInGPU}

    # load model
    net = RPCAUNet(params_net)

    device = torch.device("cuda") if CalInGPU else 'cpu'

    # device = 'cpu'
    if mfile[-3:] == 'pkl':
        state_dict = torch.load(mfile, map_location=device)
        net.load_state_dict(state_dict)
    else:
        net = torch.load(mfile)

    if CalInGPU:
        # use GPU
        net = net.cuda()
        torch.cuda.empty_cache()

    conter = Converter()
    f_number = 10
    formout = {'pres': 'concat', 'shape': (64, 64, f_number)}

    # process
    ori = []
    frameList_d = deque(maxlen=20)
    fileList = os.listdir(dicName)
    frame_count = 0
    for fileName in fileList:
        if 'ori' in fileName:
            ori.append(fileName)
        if 'IMG' in fileName:
            ori.append(fileName)
        if '.png' in fileName:
            ori.append(fileName)
    ori = sort_humanly(ori)
    frames = len(ori)
    outputSeriesL = np.zeros((512, 512, frames))
    outputSeriesS = np.zeros((512, 512, frames))

    starttime = datetime.datetime.now()

    for fileName in ori:
        img_d = mpimg.imread(dicName + '/' + fileName)
        img_d = rgb2gray(img_d)
        img_d = cv2.resize(img_d, (512, 512))
        frameList_d.append(img_d[:, :, None])
        # split patches
        if len(frameList_d) == f_number:
            videoData_d = np.concatenate(tuple(frameList_d), axis=2)
            height, width = 512, 512
            heightAxisStart = [x for x in range(0, (height - 63), 32)]
            widthAxisStart = [x for x in range(0, (width - 63), 32)]
            StartPointList = [y for y in itertools.product(heightAxisStart, widthAxisStart)]
            for h, w in StartPointList:
                videoData_torch = torch.from_numpy(videoData_d[h:h + 64, w:w + 64, :])
                outL, outS = net(to_var(videoData_torch, CalInGPU))
                [outL, outS] = conter.torch2np([outL, outS], [formout, formout])
                outputSeriesL[h:h + 64, w:w + 64, frame_count:frame_count + f_number] += outL
                outputSeriesS[h:h + 64, w:w + 64, frame_count:frame_count + f_number] += outS
                print(w, h)
            frame_count += f_number
            frameList_d.clear()
            print(frame_count)
        if frame_count > frames-f_number:
            break

    # Splice patches
    outputSeriesL[0:32, 32:480, :] = outputSeriesL[0:32, 32:480, :] / 2
    outputSeriesL[32:480, 0:32, :] = outputSeriesL[32:480, 0:32, :] / 2
    outputSeriesL[480:512, 32:480, :] = outputSeriesL[480:512, 32:480, :] / 2
    outputSeriesL[32:480, 480:512, :] = outputSeriesL[32:480, 480:512, :] / 2
    outputSeriesL[32:480, 32:480, :] = outputSeriesL[32:480, 32:480, :] / 4
    outputSeriesS[0:32, 32:480, :] = outputSeriesS[0:32, 32:480, :] / 2
    outputSeriesS[32:480, 0:32, :] = outputSeriesS[32:480, 0:32, :] / 2
    outputSeriesS[480:512, 32:480, :] = outputSeriesS[480:512, 32:480, :] / 2
    outputSeriesS[32:480, 480:512, :] = outputSeriesS[32:480, 480:512, :] / 2
    outputSeriesS[32:480, 32:480, :] = outputSeriesS[32:480, 32:480, :] / 4

    # calculate running time
    endtime = datetime.datetime.now()
    print(endtime - starttime)

    # Save output pictures
    for i in range(frames):
        
        # starttime = datetime.datetime.now()
        mask = 1 - outputSeriesS[:, :, i] # 1 - ...

        # Thresholding
        # thresh = 0.95
        # mask[mask > thresh] = 1
        # mask[mask <= thresh] = 0
        
        # mask = 1 - mask

        # if mask.ndim == 3:
        #     # Assuming mask is in BGR color format, convert it to grayscale
        #     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # # Ensure the mask is 8-bit depth
        # if mask.dtype != np.uint8:
        #     mask = mask.astype(np.uint8)
        # _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
        
        plt.imshow(mask, vmin=0, vmax=1)
        plt.axis('off')
        plt.set_cmap('gray')
        frame = plt.gca()
        frame.axes.get_yaxis().set_visible(False)
        frame.axes.get_xaxis().set_visible(False)
        plt.savefig(filePath+'/output%d.png' % i,  bbox_inches='tight', dpi=300)
        plt.close()