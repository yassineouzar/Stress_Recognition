import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2
from imutils import face_utils
import dlib

import tensorflow as tf
# Disable eager execution (Adam optimizer cannot be used if this option is enabled)
tf.compat.v1.disable_eager_execution()
tf.executing_eagerly()


from FCN8s_keras import FCN

model = FCN()
model.load_weights("Keras_FCN8s_face_seg.h5")


def vgg_preprocess(im):
    im = cv2.resize(im, (500, 500))
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:, :, ::-1]
    in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
    in_ = in_[np.newaxis, :]
    # in_ = in_.transpose((2,0,1))
    return in_


def auto_downscaling(im):
    w = im.shape[1]
    h = im.shape[0]
    while w * h >= 700 * 700:
        im = cv2.resize(im, (0, 0), fx=0.5, fy=0.5)
        w = im.shape[1]
        h = im.shape[0]
    return im


def roi_seg(image):
    im = auto_downscaling(image)
    # vgg_preprocess: output BGR channel w/ mean substracted.
    inp_im = vgg_preprocess(im)
    out = model.predict([inp_im])
    # post-process for display
    out_resized = cv2.resize(np.squeeze(out), (im.shape[1], im.shape[0]))
    out_resized_clipped = np.clip(out_resized.argmax(axis=2), 0, 1).astype(np.float64)

    mask = cv2.GaussianBlur(out_resized_clipped, (7, 7), 6)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    image = (mask[:, :, np.newaxis] * im.astype(np.float64)).astype(np.uint8)
    return image
  
  

def face_seg(path_im, path_save_im):
    list_dir = os.listdir(path_im)
    count = 0


    for i in range(int(len(list_dir))):
    #for i in range(35,140):
        list_dir1 = os.listdir(path_im + '/' +  list_dir[i])
        list_dir_save = path_save_im + '/' + list_dir[i]
        if not os.path.exists(list_dir_save):
            os.makedirs(list_dir_save)

        for j in range(int(len(list_dir1))):
            path_to_im = path_im + '/' + list_dir[i] + '/' + list_dir1[j]
            #list_dir2 = os.listdir(path_to_files)
            path_to_save_im = path_save_im + '/' +  list_dir[i] + '/' + list_dir1[j]

            if not os.path.exists(path_to_save_im):
                os.makedirs(path_to_save_im)

            img = cv2.cvtColor(cv2.imread(path_to_im), cv2.COLOR_RGB2BGR)

            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            img = roi_seg(img)
            y, x = img.shape[:2]
            h = []
            w = []
            for m in range(x):
                for l in range(y):

                    b, g, r = (img[l, m])
                    if ([b, g, r] >= [30, 30, 30]):
                        w.append(m)
                        h.append(l)
            x1, x2, y1, y2 = min(w), max(w), min(h), max(h)
            img = img[y1:y2, x1:x2]

            img = cv2.resize(img, (128, 128))

            #cv2.imwrite(path_to_save_im, img)

            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            count += 1
            print(count)

path_im = '/media/bousefsa1/My Passport/BD PPG/2 bases publiques/ubfc_phys/s1/S1'
path_save_im = '/media/bousefsa1/My Passport/BD PPG/2 bases publiques/ubfc_phys/s1_seg'

print("start")
face_seg(path_im, path_save_im)
print("finished")

