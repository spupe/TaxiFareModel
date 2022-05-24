import numpy as np
from flask import g
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from encoders import DistanceTransformer
from encoders import TimeFeaturesEncoder
from data import get_data
from data import get_test_data
from data import clean_data
from data import return_x_y
from utils import compute_rmse

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        #self.pipeline = (set_pipeline(self))
        self.X = X
        self.y = y

    def set_pipeline(self):
        # create distance pipeline
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])

        # create time pipeline
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        # create preprocessing pipeline
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        from sklearn.linear_model import LinearRegression

        # Add the model of your choice to the pipeline
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])

        return pipe

    def run(self, pipe, X_train, y_train):
        """set and train the pipeline"""
        pipe = pipe.fit(X_train,y_train)
        return pipe

    def evaluate(self, pipe, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = pipe.predict(X_test)
        return compute_rmse(y_pred, y_test)


if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    #X_train,y_train = return_x_y(df)
    X,y = return_x_y(df)
    #t = Trainer(X_train,y_train)
    t = Trainer(X,y)
    pipe = t.set_pipeline()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    df_test = get_test_data()
    df_test = clean_data(df_test)
    #X_test, y_test = return_x_y(df_test)
    pipe = t.run(pipe, X_train, y_train)
    rmse = t.evaluate(pipe,X_test,y_test)
    print(rmse)
