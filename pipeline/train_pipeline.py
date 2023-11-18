from steps.data_ingestion import IngestData
from steps.model_building import BuildModel
from steps.model_train import ModelTrain

import sys
from exception import CustomException
from logger import logging

class TrainPipeline:
        
    def trainnin_pipeline(self,train_data_path,val_data_path):
        self.ingestdata =IngestData()
        self.buildmodel = BuildModel()
        self.trainmodel = ModelTrain()
        train_data,val_data = self.ingestdata.initiate_data_ingestion(train_data_path,val_data_path)
        
        model = self.buildmodel.model_building()
        self.trainmodel.model_train(model,train_data,val_data)
        
