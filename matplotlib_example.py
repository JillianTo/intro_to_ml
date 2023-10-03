import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("iris_saved.csv")

# Line Plot
plt.plot(df['sepal_length'])
plt.title('Sepal Length Over Index')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.savefig('line_plot.png')
plt.close()

# Histogram
plt.hist(df['sepal_length'], bins=10)
plt.xticks([4,5,6,7,8])
plt.title('Histogram of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.savefig('histogram.png')
plt.close()

# Pie Chart
species_count = df['class'].value_counts()
plt.pie(species_count, labels=species_count.index,autopct='%1.1f%%')
plt.title('Species Distribution')
plt.savefig('pie_chart.png')
plt.close()

# Scatter Plot and Subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6)) # 1 row, 2 columns
