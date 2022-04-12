# using the vgg16 model as a feature extraction model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from pickle import dump
# load an image from file
image = load_img('/media/bousefsa1/My Passport/BD PPG/2 bases publiques/ubfc_phys/s1/S1/vid_s1_T1/0000.jpg', target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# load model
model = VGG16()

# remove the output layer
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
# get extracted features
features = model.predict(image)
print(features.shape)
# save to file
dump(features, open('/media/bousefsa1/My Passport/BD PPG/2 bases publiques/ubfc_phys/s1/S1/vid_s1_T1/0000.pkl', 'wb'))
