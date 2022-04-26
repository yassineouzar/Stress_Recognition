import os
import cv2
import shutil
import numpy as np
# from scipy.stats import tukey_hsd
import matplotlib.pyplot as plt


def box_plt(dataset_dir):
    list_dir = sorted(os.listdir(dataset_dir))

    for i in range(int(len(list_dir))):
        list_dir1 = os.listdir(dataset_dir + '/' + list_dir[i])
        eda_files = [filename for filename in list_dir1 if 'eda' in filename]
        eda_tasks = []
        for j in range(int(len(ippg_files))):
            eda_path = dataset_dir + '/' + list_dir[i] + '/' + eda_files[j]

            df = np.array(np.genfromtxt(ippg_path, delimiter='\n'))

            eda_tasks.append(df)

        subject = os.path.splitext(eda_path)[0][67:]

        fig, ax = plt.subplots(1, 1)

        ax.boxplot([eda_tasks[0], eda_tasks[1], eda_tasks[2]])

        ax.set_xticklabels(["meanT1", "meanT2", "meanT3"])

        ax.set_ylabel("mean" + subject)

        plt.show()

dataset_dir = ''

print("start")
box_plt(dataset_dir)
print("finished")
