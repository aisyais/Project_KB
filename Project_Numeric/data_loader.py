import pandas as pd

def load_data(data_path):
   

    return pd.read_csv(data_path)

data_path = r"D:\college\sem 3\kecerdasan buatan\project_uas\Project_Numeric\country-wise-average.csv"

data = load_data(data_path)