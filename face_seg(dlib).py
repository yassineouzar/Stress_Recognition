import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import cv2
from imutils import face_utils
import dlib


def mask_roi(image):
    p = 'shape_predictor_81_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    if detector != []:

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)
        if (rects is None):
            return ()
        result = list(enumerate(rects))
        # For each detected face, find the landmark.
        if result != []:

            for (i, rect) in enumerate(rects):
                # Make the prediction and transfom it to numpy array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                pts = np.array([shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7], shape[8],
                            shape[9], shape[10], shape[11], shape[12], shape[13], shape[14], shape[15], shape[16], shape[78],
                    shape[74], shape[79], shape[73], shape[72], shape[80], shape[71], shape[70], shape[69], shape[68],
                     shape[76], shape[75], shape[77]], np.int32)

                mask = np.zeros_like(image)
                mask = cv2.fillPoly(mask, [pts], (255, 255, 255))
                image = cv2.bitwise_and(image, mask)
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
            path_to_save_im = path_save_im + '/' +  list_dir[i] + '/' + list_dir1[j]

            if not os.path.exists(path_to_save_im):
                os.makedirs(path_to_save_im)

            img = cv2.imread(path_to_im)

            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            img = mask_roi(img)
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
