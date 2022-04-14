import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import matplotlib
import sklearn
import tensorflow as tf
import numpy as np
import scipy.io
import sys
import argparse
import csv

import plotly.graph_objects as go

from model_MTTS_CAN import Attention_mask, MTTS_CAN
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter
from inference_preprocess import preprocess_raw_video, detrend
from scipy.signal import find_peaks, stft, lfilter, butter, welch, resample
import matplotlib.pyplot as plt
import pandas as pd

def predict_vitals(vid_dir, sampling_rate, batch_size, path_to_save_ippg, path_GT_BVP):
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    model_checkpoint = 'mtts_can.hdf5'
    #batch_size = args.batch_size
    fs = sampling_rate
    #sample_data_path = "G:/F001_T1.mkv"
    sample_data_path = vid_dir

    dXsub = preprocess_raw_video(sample_data_path, dim=36)
    #print('dXsub shape', dXsub.shape)

    dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
    dXsub = dXsub[:dXsub_len, :, :, :]
    #print("video", dXsub)
    print("dxsublen", dXsub_len)

    model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model.load_weights(model_checkpoint)

    f = pd.read_csv(path_GT_BVP)

    secs = len(f) / 64  # Number of seconds in signal X
    samps = secs * 35  # Number of samples to downsampqle
    gtData = resample(f, int(round(samps)), domain='time')

    gtData2 = np.array(gtData)
    print("GT", len(gtData2))
    gtFloat = gtData2.astype(np.float)

    gtPRvalues = []
    estPRvalues = []

    # Calculating HR rate without SNR formulas
    # we used the estimated values and ground truth values obtained on thirty-second video segments with the starting
    # points at one-second intervals for each video
    for i in range(0, dXsub_len - 6300, 15):
        print("finished")

        yptest = model.predict((dXsub[i:i+6300, :, :, :3], dXsub[i:i+6300, :, :, -3:]), batch_size=batch_size, verbose=0)
        gtPR = gtFloat[i:i+6300]
        gtPRmean = np.mean(gtPR)
        gtPRvalues = np.append(gtPRvalues, gtPRmean)
        # print(gtPRvalues)
        print("********************************")

        pulse_pred = yptest[0]


        pulse_pred = detrend(np.cumsum(pulse_pred), 100)
        [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))
        #print(path_to_save_ippg)
        #print(pulse_pred)
        pulse_pred.tofile(path_to_save_ippg, sep='\n')

def vid2ippg(dataset_dir, sampling_rate, batch_size):

    list_dir = sorted(os.listdir(dataset_dir))
    extension = ".zip"
    #for i in list_dir:
    for i in range(34, len(list_dir)):
        if not list_dir[i].endswith(extension):  # check for ".zip" extension
            path_to_vid = os.path.join(dataset_dir, list_dir[i])
            #print(path_to_vid)
            vid_files = [filename for filename in sorted(os.listdir(path_to_vid))if filename.endswith(".avi")]

            for vid in vid_files:
                vid_dir = os.path.join(path_to_vid, vid)
                path_to_save_ippg = (os.path.splitext(vid_dir)[0] + ".csv").replace("vid", "ippg")
                path_GT_BVP = (os.path.splitext(vid_dir)[0] + ".csv").replace("vid", "bvp")
                #print(path_GT_BVP, path_to_save_ippg)
                predict_vitals(vid_dir, sampling_rate, batch_size, path_to_save_ippg, path_GT_BVP)


dataset_dir = '/media/bousefsa1/My Passport/BD PPG/2 bases publiques/ubfc_phys'
batch_size = 10
sampling_rate = 35

vid2ippg(dataset_dir, sampling_rate, batch_size)
