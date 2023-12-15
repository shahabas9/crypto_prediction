import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
import sys
from src.utils import save_object


@dataclass
class Datatransformation_config:
    preprocessor_obj_file_path=os.path.join("Artifacts","preprocessor.pkl")

class Datatransformation:
    def __init__(self):
        self.data_transformation_config=Datatransformation_config()
    
    def get_data_transformation_obj(self):
        try:
            logging.info("data transformation initiated")
            # numerical and categorical columns were separated
            numerical_columns=['carat', 'depth', 'table', 'x', 'y', 'z']
            categorical_columns=['cut', 'color', 'clarity']

            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("pipeline initiated")
            # numerical pipeline
            num_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ("scaler",StandardScaler())
                ]
            )


            # categorical pipeline
            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ("scaler",StandardScaler())
                ]
            )

            # joining numerical and categorical pipeline
            preprocessor=ColumnTransformer([
                ("numerical_pipeline",num_pipeline,numerical_columns),
                ("categorical_pipeline",cat_pipeline,categorical_columns)
            ])

            return preprocessor
            logging.info("pipeline completed")


            
        except Exception as e:
            logging.info("Exception occured at data transformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("Reading train and test data started")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test completed")
            logging.info(f"Train DataFrame head :\n{train_df.head().to_string()}")
            logging.info(f"Test DataFrame head :\n{test_df.head().to_string()}")
            logging.info(" Getting preprocessor object")
            preprocessor_obj=self.get_data_transformation_obj()

            target_column_name="price"
            drop_columns=[target_column_name,'id']

            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            logging.info("Applying preprocessing on train and test data")

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            

            save_object(
                file_path=Datatransformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info("preprocessor pickle file saved")

            return(
                train_arr,test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            logging.info("Exception occured at initiate_data_transformation")
            raise CustomException(e,sys)
