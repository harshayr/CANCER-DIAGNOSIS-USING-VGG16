import pandas as pd
import numpy as np
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator   # for lebelling images
from keras.preprocessing import image

import sys
from exception import CustomException
from logger import logging



class IngestData:
    # def __init__(self, train_data_path:str, val_data_path:str):
    #     self.train_data_path = train_data_path
    #     self.val_data_path = val_data_path

    def initiate_data_ingestion(self,train_data_path,val_data_path):

        try:
            logging.info("Data ingestion stared ")

            train = ImageDataGenerator(rescale = 1/255)
            val = ImageDataGenerator(rescale=1/255)

            train_data = train.flow_from_directory(train_data_path, target_size = (224,224),
                                                batch_size = 30,class_mode = 'categorical')
            val_data = val.flow_from_directory(val_data_path, target_size = (224,224),
                                            batch_size = 30,class_mode = 'categorical')
            logging.info("Data ingestion complete and returns  train_data, val_data")
        except Exception as e:
            raise CustomException(e,sys)

        return train_data,val_data


