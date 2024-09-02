import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import(
    
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import custom_exception
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class modelTrainerConfig:
    trained_model_path=os.path.join('artifacts','model.pkl')

class modelTrainer:
    def __init__(self):
        self.model_trainer_config=modelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "Kneighbours regressor":KNeighborsRegressor(),
                "Xgboost Regressor":XGBRegressor(),
                "Adaboost Regressor":AdaBoostRegressor(),
                "Catboost Regressor": CatBoostRegressor(logging_level='Silent')
            }


            model_report:dict=evaluate_models(X_train=x_train,Y_train=y_train,X_test=x_test,Y_test=y_test,models=models)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise custom_exception("didnt found any best model")
            logging.info(f"best model is {best_model}")

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )
            predicted=best_model.predict(x_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise custom_exception(e,sys)    