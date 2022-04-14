import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# example of using the vgg16 model as a feature extraction model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from pickle import dump


# load model
model = VGG16()
# remove the output layer
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
def Features_extraction(dataset_dir, path_pkl):
    list_dir = sorted(os.listdir(dataset_dir))
    for i in range(int(len(list_dir))):
        # for i in range(35,140):
        list_dir1 = sorted(os.listdir(dataset_dir + '/' + list_dir[i]))



        for j in range(int(len(list_dir1))):
            path_to_im = dataset_dir + '/' + list_dir[i] + '/' + list_dir1[j]
            pkl_dir_save = path_pkl + '/' + list_dir[i] + '/' + list_dir1[j]
            if not os.path.exists(pkl_dir_save):
                os.makedirs(pkl_dir_save)
            for im in sorted(os.listdir(path_to_im)):
                im_path = path_to_im + '/' + im
                path_to_save_pkl = (path_pkl + '/' + list_dir[i] + '/' + list_dir1[j] + '/' + im).replace("jpg", "pkl")

                # load an image from file
                image = load_img(im_path, target_size=(224, 224))
                # convert the image pixels to a numpy array
                image = img_to_array(image)
                # reshape data for the model
                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                # prepare the image for the VGG model
                image = preprocess_input(image)
                # get extracted features
                features = model.predict(image)
                # save to file
                dump(features, open(path_to_save_pkl, 'wb'))


dataset_dir = '/media/bousefsa1/My Passport/BD PPG/2 bases publiques/ubfc_phys_frames'
path_pkl = '/media/bousefsa1/My Passport/BD PPG/2 bases publiques/ubfc_phys_pkl'

print("start")
Features_extraction(dataset_dir, path_pkl)
print("finished")
