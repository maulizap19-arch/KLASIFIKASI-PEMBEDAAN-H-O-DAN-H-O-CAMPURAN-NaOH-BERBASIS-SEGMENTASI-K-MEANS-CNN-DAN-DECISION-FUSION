import numpy as np
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_pre
from tensorflow.keras.applications.densenet import preprocess_input as dense_pre
from tensorflow.keras.applications.inception_v3 import preprocess_input as inc_pre
import cv2

def preprocess_image(img, model_type):
    img = cv2.resize(img, (224, 224))

    if model_type == "vgg":
        return vgg_pre(img.astype(np.float32))
    elif model_type == "densenet":
        return dense_pre(img.astype(np.float32))
    elif model_type == "inception":
        return inc_pre(img.astype(np.float32))
