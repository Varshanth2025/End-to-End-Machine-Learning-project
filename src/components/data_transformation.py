from sklearn.preprocessing import StandardScaler,OneHotEncoder
import pandas as pd
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import custom_exception
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class dataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class dataTransformation:
    def __init__(self):
        self.data_transformation_config=dataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numerical_columns=['writing score','reading score']
            categorical_columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))

                ]
            )    
            logging.info("numerical and categorical transformation is done")
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("categorical_piepline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise custom_exception(e,sys)
        
    def initiate_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data completed")
            logging.info("obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_obj()

            target_column_name='math score'
            numerical_columns=['reading score', 'writing score']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"applying preprocessing object on train data and test data")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info(f"saved preprocessing object.")

            save_object(
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessing_obj
            )

        

            return (
                train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise custom_exception(e,sys)
        