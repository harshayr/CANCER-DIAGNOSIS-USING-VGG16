import tensorflow as tf
import keras
from keras.applications.vgg16 import VGG16
from keras.models import Model

import sys
from exception import CustomException
from logger import logging
from keras.layers import Flatten,Dense


class BuildModel:
    def model_building(self):
        try:
            logging.info("Model building started ")

            vgg16 = VGG16(input_shape = [224,224,3], weights = 'imagenet', include_top = False )

            for layer in vgg16.layers:
                layer.trainable = False
            x = Flatten()(vgg16.output)
            prediction = Dense(3,activation = 'softmax')(x)
            model = Model(inputs = vgg16.input, outputs = prediction)

            logging.info("Model building completed and returns model ")
        except Exception as e:
            raise CustomException(e,sys)
            
        return model
   