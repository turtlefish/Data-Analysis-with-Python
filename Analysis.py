from logging.handlers import RotatingFileHandler
from tokenize import group
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


other_path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df = pd.read_csv(other_path)

# var = "drive-wheels"
# print(df[[var, "price"]].corr())
# sns.boxplot(x=var, y="price", data=df)
# plt.show()


drive_wheel_counts = df["drive-wheels"].value_counts().to_frame()
drive_wheel_counts.rename(columns={"drive-wheels": "value_counts"}, inplace=True)
drive_wheel_counts.index.name = "drive-wheels"

engine_loc_counts = df["engine-location"].value_counts().to_frame()
engine_loc_counts.rename(columns={"engine-location": "value_counts"}, inplace=True)
engine_loc_counts.index.name = "engine-location"


df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
# print(grouped_pivot)

df_gptest2 = df[["body-style", "price"]]
grouped_test2 = df_gptest2.groupby(["body-style"], as_index=False).mean()
# print(grouped_test2)

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap="RdBu")

row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

fig.colorbar(im)
plt.show()