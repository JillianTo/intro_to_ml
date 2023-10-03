import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("iris_saved.csv")

plt.plot(df['sepal_length'])
plt.title('Sepal Length Over Index')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.savefig('line_plot.png')
