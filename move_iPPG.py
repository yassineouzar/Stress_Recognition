import os
import cv2
import shutil

def move_ippg(dataset_dir, destination_path):

        list_dir = sorted(os.listdir(dataset_dir))

    for i in range(int(len(list_dir))):
        list_dir1 = os.listdir(dataset_dir + '/' + list_dir[i])
        ippg_files = [filename for filename in list_dir1 if 'bvp' in filename]
        for j in range(int(len(ippg_files))):
            ippg_path = dataset_dir + '/' + list_dir[i] + '/' + ippg_files[j]
            path_to_save = destination_path + '/' + os.path.splitext(ippg_path)[0][64:]
            print(path_to_save)
            # if not os.path.exists(path_to_save):
            #     os.makedirs(path_to_save)
            destination = (path_to_save + '/' + os.path.splitext(ippg_path)[0][-11:] + os.path.splitext(ippg_path)[1]).replace("//","/")

            # Copy the content of
            # source to destination

            try:
                shutil.copyfile(ippg_path, destination)
                print("File copied successfully.")

            # If source and destination are same
            except shutil.SameFileError:
                print("Source and destination represents the same file.")

            # If destination is a directory.
            except IsADirectoryError:
                print("Destination is a directory.")

            # If there is any permission issue
            except PermissionError:
                print("Permission denied.")

            # For other errors
            except:
                print("Error occurred while copying file.")
                
    
dataset_dir= '/media/bousefsa1/My Passport/BD PPG/2 bases publiques/ubfc_phys'
destination_path = '/media/bousefsa1/My Passport/BD PPG/2 bases publiques/ubfc_phys_ippg'

print("start")
move_ippg(dataset_dir, destination_path)
print("finished")
