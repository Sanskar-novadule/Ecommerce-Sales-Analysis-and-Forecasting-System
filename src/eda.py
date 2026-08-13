import pandas as pd 

PROCESSED_DATA_PATH = "data/sales_data_processed.csv"

df = pd.read_csv(PROCESSED_DATA_PATH)
print(df.head())
