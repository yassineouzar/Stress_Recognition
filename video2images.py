import os
import cv2

def Frames_extraction(path_video, path_save_images):
    list_dir = sorted(os.listdir(path_video))
    count = 0
    videos = [filename for filename in list_dir if filename.endswith("avi")]
    for i in range(int(len(videos))):
        vid_dir = path_video + '/' +  videos[i]
        path_to_save = path_save_images + '/' +  os.path.splitext(vid_dir)[0][64:]

        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        cap = cv2.VideoCapture(vid_dir)

        counter = 0
        while (cap.isOpened()):
            fps = cap.get(cv2.CAP_PROP_FPS)
            ret, frame = cap.read()

            print(fps)
            if ret == False:
                break
            name = "/%.4d" % counter + '.jpg'
            name = path_to_save + name
            print(name)

            #display images
            # cv2.imshow('img', frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # save the extracted frames
            cv2.imwrite(name, frame)
            counter += 1
            print(counter)

        cap.release()
        cv2.destroyAllWindows()


path_video= '/media/bousefsa1/My Passport/BD PPG/2 bases publiques/ubfc_phys/s1'
path_save_images = '/media/bousefsa1/My Passport/BD PPG/2 bases publiques/ubfc_phys/s1/S1'

print("start")
Frames_extraction(path_video, path_save_images)
print("finished")
