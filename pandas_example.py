import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv(url, header=None, names=column_names)

df.to_csv("iris_saved.csv", index=False)
df = pd.read_csv("iris_saved.csv")

print("Shape of the DataFrame: ", df.shape)
print("\nFirst 5 rows of the DataFrame: ")
print(df.head())
print("\nData types of each column:")
print(df.dtypes)
print("\nSummary statistics of the DataFrame:")
print(df.describe())
mean_sepal_length = df['sepal_length'].mean()
median_sepal_width = df['sepal_width'].median()
std_petal_length = df['petal_length'].std()
grouped_mean = df.groupby('class').mean()
grouped_stats = df.groupby('class').agg(['mean','std','min','max'])
class_count = df['class'].value_counts()
missing_values = df.isnull().sum()
df.drop_duplicates(inplace=True)
