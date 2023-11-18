import tensorflow as tf
import keras
import os
import sys
from dataclasses import dataclass


from keras.models import load_model 
from keras.losses import BinaryCrossentropy
from keras.optimizers.legacy import Adam 
from keras.metrics import RootMeanSquaredError, BinaryAccuracy,FalsePositives,FalseNegatives, Precision, AUC ,Recall,TruePositives ,TrueNegatives
from keras.models import Model
from keras.layers import Input, Layer,Dropout
from keras.regularizers import L1,L2
from keras.callbacks import Callback, CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard
import sys
from exception import CustomException
from logger import logging

# from steps.model_building import BuildModel
@dataclass
class ModelTrainConfig:
    model_path = os.path.join("artifacts","cancer_model.h5")

class ModelTrain:
    def __init__(self):
        self.path = ModelTrainConfig()
        # self.train_data = train_data
        # self.val_data = val_data

    def model_train(self,model,train_data,val_data):
        try:
            metrics = [BinaryAccuracy (name= 'accuracy'), Precision(name='precision'),Recall(name= 'recall')]
            csv_callback = CSVLogger('csv_logger_cancer',
                            separator = ',', append = False)
            tensorboard_callback = TensorBoard(log_dir="logs/")

            early_callback = EarlyStopping(monitor = 'val_loss', min_delta = 0,patience = 3, verbose = 1, mode = 'auto',baseline = None,restore_best_weights = False )
            
            model.compile(loss = BinaryCrossentropy(),optimizer = Adam(learning_rate = 0.001),metrics = metrics)
            logging.info("Model compiling done")
            model.fit_generator( train_data , validation_data = val_data , epochs = 10,verbose = 1, callbacks = [tensorboard_callback,early_callback])
            logging.info("Model Training started")
            
            dir_path = os.path.dirname(os.path.join("artifacts","cancer_model.h5"))
            os.makedirs(dir_path,exist_ok=True)
            logging.info("Model Training done")
            
            model.save(self.path.model_path)
            logging.info("Model save succesfuly")
        except Exception as e:
            raise CustomException(e,sys)
        
        return None
        

