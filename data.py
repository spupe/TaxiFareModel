import pandas as pd

AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"


def get_data(nrows=10_000):
    '''returns a DataFrame with nrows from s3 bucket'''
    # url not working, resorting to local data
    df = pd.read_csv('raw_data/taxi-fare-train.csv', nrows=nrows)
    return df

def get_test_data():
    '''returns a DataFrame with nrows from s3 bucket'''
    # url not working, resorting to local data
    df = pd.read_csv('raw_data/taxi-fare-test.csv')
    return df

def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df

def return_x_y(df):
    y = df.pop("fare_amount")
    X = df
    return X,y


if __name__ == '__main__':
    df = get_data()
