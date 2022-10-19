import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib import pyplot

web_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(web_path, names=headers)

df.replace("?", np.nan, inplace=True)

missing_data = df.isnull()


avg_norm_loss = df["normalized-losses"].astype(float).mean(axis=0)
avg_stroke = df["stroke"].astype(float).mean(axis=0)
avg_bore = df["bore"].astype(float).mean(axis=0)
avg_horsepower = df["horsepower"].astype(float).mean(axis=0)
avg_peak_rpm = df["peak-rpm"].astype(float).mean(axis=0)

df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)
df["stroke"].replace(np.nan, avg_stroke, inplace=True)
df["bore"].replace(np.nan, avg_bore, inplace=True)
df["horsepower"].replace(np.nan, avg_horsepower, inplace=True)
df["peak-rpm"].replace(np.nan, avg_peak_rpm, inplace=True)

# print(df.loc[df["num-of-doors"].isnull()])
# print(df["num-of-doors"].value_counts())

df["num-of-doors"].replace(np.nan, "four", inplace=True)
df.dropna(subset=["price"], axis=0, inplace=True)

df.reset_index(drop=True, inplace=True)

df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

df['city-L/100km'] = 235/df["city-mpg"]
df["highway-mpg"] = 235/df["highway-mpg"]

df.rename(columns={"highway-mpg": "highway-L/100km"}, inplace=True)

df['length'] = df['length'] / df['length'].max()
df['width'] = df['width'] / df['width'].max()
df["height"] = df["height"] / df["height"].max()

# print(df[["length","width","height"]].head())

df["horsepower"]=df["horsepower"].astype(int, copy=True)

# plt.pyplot.hist(df["horsepower"])

# # set x/y labels and plot title
# plt.pyplot.xlabel("horsepower")
# plt.pyplot.ylabel("count")
# plt.pyplot.title("horsepower bins")
# plt.pyplot.hist(df["horsepower"])
# # set x/y labels and plot title
# plt.pyplot.xlabel("horsepower")
# plt.pyplot.ylabel("count")
# plt.pyplot.title("horsepower bins")


bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True)

dummy_var_1 = pd.get_dummies(df["fuel-type"])

dummy_var_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
dummy_var_1.head()

# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_var_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)

dummy_var_2 = pd.get_dummies(df["aspiration"])
dummy_var_2.rename(columns={"std":"aspiration-std", "turbo": "aspiration-turbo"}, inplace=True)

df = pd.concat([df, dummy_var_2], axis=1)
df.drop("aspiration", axis=1, inplace=True)

print(df.head())

df.to_csv('clean_df.csv')