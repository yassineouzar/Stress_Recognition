#python code/predict_vitals.py --video_path "test-video/esra.mp4"
import matplotlib
import sklearn
import tensorflow as tf
import numpy as np
import scipy.io
import os
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

size=18
params = {'legend.fontsize': 'large',
          'figure.figsize': (20,8),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
          'axes.titlepad': 25}
plt.rcParams.update(params)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


#plt.tick_params(labelsize=16)
def predict_vitals(args):
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    model_checkpoint = 'mtts_can.hdf5'
    batch_size = args.batch_size
    fs = args.sampling_rate
    sample_data_path = "/media/bousefsa1/My Passport/BD PPG/2 bases publiques/ubfc_phys/s1/vid_s1_T1.avi"

    dXsub = preprocess_raw_video(sample_data_path, dim=36)

    dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
    dXsub = dXsub[:dXsub_len, :, :, :]
    print("dxsublen", dXsub_len)

    model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model.load_weights(model_checkpoint)

    f = pd.read_csv("/media/bousefsa1/My Passport/BD PPG/2 bases publiques/ubfc_phys/s1/bvp_s1_T1.csv")

    secs = len(f) / 64  # Number of seconds in signal X
    samps = secs * 35  # Number of samples to downsampqle
    gtData = resample(f, int(round(samps)), domain='time')

    gtData2 = np.array(gtData)
    print("GT", len(gtData2))
    gtFloat = gtData2.astype(np.float)

    gtFloat_norm= NormalizeData(gtFloat[0:6000])

    gtPRvalues = []
    estPRvalues = []

    # Calculating HR rate without SNR formulas
    # we used the estimated values and ground truth values obtained on thirty-second video segments with the starting
    # points at one-second intervals for each video
    for i in range(0, dXsub_len - 6000, 15):
        print("finished")

        yptest = model.predict((dXsub[i:i+6000, :, :, :3], dXsub[i:i+6000, :, :, -3:]), batch_size=batch_size, verbose=0)
        gtPR = gtFloat[i:i+6000]
        gtPRmean = np.mean(gtPR)
        gtPRvalues = np.append(gtPRvalues, gtPRmean)
        # print(gtPRvalues)
        print("********************************")

        pulse_pred = yptest[0]

        pulse_pred = detrend(np.cumsum(pulse_pred), 100)
        [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))

        pulse_pred_norm = NormalizeData(pulse_pred)

        # Peaks detection
        indexes_peaks, _ = scipy.signal.find_peaks(pulse_pred_norm, distance=10, height=0.05)
        indexes_peaks1, _ = scipy.signal.find_peaks(gtFloat_norm.ravel(), distance=10, height=0.05)

        plt.plot(indexes_peaks, pulse_pred_norm[indexes_peaks], 'r.', linewidth=5, markersize=10, label="Peaks")
        plt.plot(indexes_peaks1, gtFloat_norm[indexes_peaks1], 'r.', linewidth=5, markersize=10)

        plt.plot(gtFloat_norm, 'g', label="Ground truth PPG")
        plt.plot(pulse_pred_norm, 'b', label="Recovered signal by MTTS-CAN",linewidth=2)
        plt.xlabel('Frames', fontsize = 18)
        plt.ylabel('Recovered iPPG signal', fontsize = 18)
        plt.legend()
        plt.show()

        pxx, frequency = matplotlib.pyplot.psd(pulse_pred_norm, NFFT=len(pulse_pred_norm), Fs=35, window=np.hamming(len(pulse_pred)))

        LL_PR = 40  #lower limit pulse rate
        UL_PR = 240


        FMask = (frequency >= (LL_PR / 60)) & (frequency <= (UL_PR / 60))

        FRange = frequency[FMask]
        PRange = pxx[FMask]
        MaxInd = np.argmax(PRange)
        PR_F = FRange[MaxInd]
        PR = PR_F * 60
        estPRvalues = np.append(estPRvalues, PR)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='processed video path')
    parser.add_argument('--sampling_rate', type=int, default = 35, help='sampling rate of your video')
    parser.add_argument('--batch_size', type=int, default = 100, help='batch size (multiplier of 10)')
    args = parser.parse_args()

    predict_vitals(args)
