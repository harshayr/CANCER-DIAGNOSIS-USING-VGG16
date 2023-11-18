from pipeline.train_pipeline import TrainPipeline

if __name__ == "__main__":
    run_pipeline = TrainPipeline()
    run_pipeline.trainnin_pipeline("/Users/harshalrajput/Desktop/MLOPS_cancer_project/dataset/training_set",
                                   "/Users/harshalrajput/Desktop/MLOPS_cancer_project/dataset/validation_set")
    

