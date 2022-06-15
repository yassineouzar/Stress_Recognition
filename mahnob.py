# -- Modules and packages to import for demo
import os.path

from pyVHR.datasets.dataset import datasetFactory
# from pyVHR.methods.base import methodFactory
import numpy as np
from pyVHR.plot import visualize_ECGs
import matplotlib.pyplot as plt

data_dir = '/media/bousefsa1/My Passport/BD PPG/2 bases publiques/MAHNOB/Sessions/'

# -- dataset object
dataset = datasetFactory("MAHNOB", videodataDIR=data_dir, BVPdataDIR=data_dir)

# -- videos filenames of the dataset
#print("List of video filenames:")
print(*dataset.videoFilenames, sep="\n")


for i in range(len(dataset.sigFilenames)):
    # -- ground-truth (GT) signal
    fname = dataset.getSigFilename(i)
    # pulse_rate_name = fname.replace('.bdf', '.txt')
    # pulse_rate_name = 'ECG.txt'
    # print(os.path.basename(os.path.normpath(os.path.splitext(fname)[0])))
    # print(os.path.dirname(os.path.normpath(os.path.splitext(fname)[0])))
    pulse_rate_name = os.path.join(os.path.dirname(os.path.normpath(os.path.splitext(fname)[0])), 'ECG.txt')
    # print(pulse_rate_name)
    # -- load signal and build a BVPsignal or ECGsignal object
    sigGT = dataset.readSigfile(fname)
    # -- plot ECG signal
    # plt.plot(-sigGT.data[0])
    # plt.show()
    # -- get HR & peaks time
    # sigGT.getBPM(winsize=5)
    with open(pulse_rate_name, 'w') as file:
        for item in -sigGT.data[0]:
            file.write("%s \n" % item)

"""

# -- GT signal filenames (ECG or BVP) of the dataset
print("\nList of GT signal filenames:")
print(*dataset.sigFilenames, sep="\n")

# -- ground-truth (GT) signal
idx = 2   # index of signal within the list dataset.videoFilenames
fname = dataset.getSigFilename(idx)

# -- load signal and build a BVPsignal or ECGsignal object
sigGT = dataset.readSigfile(fname)
# sigGT.plot()
# visualize_ECGs(sigGT)

# -- plot signal + peaks
#sigGT.findPeaks(distance=30)
# sigGT.getBPM(winsize=5)
plt.plot(-sigGT.data[0])
plt.show()
# sigGT.visualize_ECGs(sigGT, window=5)

vhr.plot.visualize_ECGs(patch_bvps, w)

for i in range(len(dataset.sigFilenames)):
    # -- ground-truth (GT) signal
    fname = dataset.getSigFilename(i)
    pulse_rate_name = fname.replace('.bdf', '.txt')
    # -- load signal and build a BVPsignal or ECGsignal object
    sigGT = dataset.readSigfile(fname)
    # -- plot signal + peaks
    sigGT.getBPM(winsize=5)

    # -- fix the window size for BPM estimate
    winSizeGT = 5
    bpmGT, timesGT = sigGT.getBPM(winSizeGT)
    # print("BPMs of the GT signal averaged on winSizeGT = 7 sec")
    #print(bpmGT)
    #print(len(bpmGT))
    PR_p = []

    for n in range(len(bpmGT)):
        a = bpmGT[n]
        b = round((60 / a) * 25)
        vect_p = [bpmGT[n]]
        PRi_p = np.repeat(vect_p, b)
        PR_p.extend(PRi_p)
    print(len(PR_p))
    # with open(pulse_rate_name, 'w') as file:
    #     for item in PR_p:
    #         file.write("%s \n" % item)



# -- ground-truth (GT) signal
idx = 1   # index of signal within the list dataset.videoFilenames
fname = dataset.getSigFilename(idx)

# -- load signal and build a BVPsignal or ECGsignal object
sigGT = dataset.readSigfile(fname)
#sigGT.plot()

# -- plot signal + peaks
#sigGT.findPeaks(distance=30)
sigGT.getBPM(winsize=5)

#sigGT.plot()

# -- fix the window size for BPM estimate
winSizeGT = 5
bpmGT, timesGT = sigGT.getBPM(winSizeGT)
#print("BPMs of the GT signal averaged on winSizeGT = 7 sec")
print(bpmGT)
print(len(bpmGT))
PR_p = []

for n in range(len(bpmGT)):
    a = bpmGT[n]
    b = round((60 / a) * 25)
    vect_p = [bpmGT[n]]
    PRi_p = np.repeat(vect_p, b)
    PR_p.extend(PRi_p)
print(len(PR_p))
# -- plot spectrogram
#sigGT.displaySpectrum()


    path_to_save = 
    with open(path_to_save, 'w') as file:
        for item in iHR25:
            file.write("%s \n" % item)
"""