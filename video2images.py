import os
import cv2

def get_im_train(path_im, path_save_im):
    list_dir = sorted(os.listdir(path_im))
    count = 0

    for i in range(int(len(list_dir))):
        vid_dir = path_im + '/' +  list_dir[i]
        dir_save1 = path_save_im + '/' +  list_dir[i]
        path_to_save = os.path.splitext(dir_save1)[0] + '/' + os.path.splitext(dir_save1)[0][-8:]
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        cap = cv2.VideoCapture(vid_dir)

        j = 0
        frame_rate=25
        counter = 0
        while (cap.isOpened()):
            fps = cap.get(cv2.CAP_PROP_FPS)
            ret, frame = cap.read()

            print(fps)
            if ret == False:
                break
            name = "/%.4d" % j + '.jpg'
            name = path_to_save + name
            print(name)

            #cv2.imshow('img', frame)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            # writing the extracted images
            cv2.imwrite(name, frame)
            # cv2.imwrite('kang'+str(i)+'.jpg',frame)
            j += 1


        cap.release()
        cv2.destroyAllWindows()

path_im = 'G:/V4V/videos'
path_save_im = 'G:/V4V/videos'

print("begin1")
get_im_train(path_im, path_save_im)
print("finished")


"""

# Read the video from specified path
url = "/media/bousefsa1/Elements/BD PPG/2 bases publiques/V4V/test/Phase 2_ Testing set/Videos/Test/10219632.mkv"
cap = cv2.VideoCapture(url)

url_save = "/home/ouzar1/Desktop/Dataset1/model/ROI_ubfc_dlib/F003/T2"
url_save = "/media/bousefsa1/Elements/BD PPG/2 bases publiques/V4V/test/Phase 2_ Testing set/Frames/"
i = 0
frame_rate=25
counter = 0
while (cap.isOpened()):
    cap.set(cv2.CAP_PROP_FPS, 5)
    ret, frame = cap.read()

    #print(fps)
    if ret == False:
        break
    name = "%.4d " % i + '.jpg'
    name = url_save + name
    print(name)

    #cv2.imshow('img', frame)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # writing the extracted images
    #cv2.imwrite(name, frame)
    # cv2.imwrite('kang'+str(i)+'.jpg',frame)
    #i += 1


cap.release()
cv2.destroyAllWindows()

"""
"""
import cv2

# Read the video from specified path
url = "/home/ouzar1/Desktop/Dataset1/model/ubfc_resampled/3.avi"
cap = cv2.VideoCapture(url)

url_save = "/home/ouzar1/Desktop/Dataset1/model/ROI_ubfc_dlib/F003/T1"
i = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)

    #print(fps)
    if ret == False:
        break
    name ='/frame' + str(i) + '.jpg'
    name = url_save + name
    print(name)
    # writing the extracted images
    #cv2.imwrite(name, frame)
    # cv2.imwrite('kang'+str(i)+'.jpg',frame)
    i += 1

cap.release()
cv2.destroyAllWindows()
"""
