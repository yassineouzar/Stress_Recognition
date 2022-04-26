import os
import cv2
import shutil
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

def bandpass_butter(arr, cut_low, cut_high, rate, order=2):
    nyq = 0.5 * rate
    low = cut_low / nyq
    high = cut_high / nyq
    b, a = signal.butter(order, [low, high], btype='band', analog=False, output='ba')
    out = signal.filtfilt(b, a, arr)
    return out


def Compute_HR(dataset_dir):
    list_dir = sorted(os.listdir(dataset_dir))

    for i in range(int(len(list_dir))):
        list_dir1 = os.listdir(dataset_dir + '/' + list_dir[i])
        ippg_files = [filename for filename in list_dir1 if 'bvp' in filename]
        mean = []
        std = []
        for j in range(int(len(ippg_files))):
            ippg_path = dataset_dir + '/' + list_dir[i] + '/' + ippg_files[j]

            bvp = np.array(np.genfromtxt(ippg_path, delimiter='\n'))
            sig_filtered = bandpass_butter(bvp, 0.6, 4, 64, 2)
            nbr_pt = 30
            v = np.ones((1, round(nbr_pt / 2)))
            b = signal.convolve(v, v)
            a = sum(b)
            b = b.reshape((-1,))
            sig_liss = signal.filtfilt(b, a, sig_filtered)
            # plt.plot(bvp)
            # plt.plot(sig_filtered)
            plt.plot(sig_liss, label="Sig_liss")
            # plt.show()

            sampling_rate = 64
            time = np.arange(0, ((len(sig_liss) / sampling_rate) - 1 // sampling_rate), 1 / sampling_rate)


            #detection des pics et des vallies
            indexes_peaks, _ = signal.find_peaks(-sig_liss, distance=10, height=0.5)
            indexes_valleys, _ = signal.find_peaks(sig_liss, distance=10, height=0.5)

            plt.plot(indexes_peaks, sig_liss[indexes_peaks], 'rx', linewidth=5, markersize=5, label="Peaks")
            plt.plot(indexes_valleys, sig_liss[indexes_valleys], 'mx', linewidth=5, markersize=5, label="Valleys")
            plt.legend()
            plt.show()
            iHR_peaks = np.gradient(time[indexes_peaks])
            iHR_valleys = np.gradient(time[indexes_valleys])

            iHR_peaks = 60 / iHR_peaks
            iHR_valleys = 60 / iHR_valleys
            hr_p = []
            hr_v = []

            for n in range(len(iHR_valleys)):
                a = iHR_valleys[n]
                b = round((60 / a) * sampling_rate)
                vect_v = [iHR_valleys[n]]
                hri_v = np.repeat(vect_v, b)
                hr_v.extend(hri_v)

            for n in range(len(iHR_peaks)):
                a = iHR_peaks[n]
                b = round((60 / a) * sampling_rate)
                vect_p = [iHR_peaks[n]]
                hri_p = np.repeat(vect_p, b)
                hr_p.extend(hri_p)
            plt.plot(hr_p, label="HR peaks")
            plt.plot(hr_v, label="HR valleys")
            plt.legend()

            plt.show()


dataset_dir = '/media/bousefsa1/My Passport/BD PPG/2 bases publiques/ubfc_phys'

print("start")
Compute_HR(dataset_dir)
print("finished")
