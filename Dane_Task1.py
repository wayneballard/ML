import numpy as np
import pandas as pd
import seaborn as sns #statistical data visualization library
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

"""DATA EVALUATION"""
df = pd.read_csv("https://raw.githubusercontent.com/wayneballard/ML/refs/heads/main/International_Education_Costs.csv")
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
print(f"First 10 elements of DataFrame:\n {df.head(10)}")
print(f"Simple Statistics of DataFrame:\n {df.describe()}")
print(f"DataTypes of DataFrame:\n {df.dtypes}")
print(f"Size of DataFrame is {df.shape[0]} rows and {df.shape[1]} columns")
df = df.rename(columns={col: col.lower() for col in df.columns})
columns = df.columns.tolist()
print(f"The columns are:\n {columns}")


"""DATA VISUALIZATION"""
#Checking for missing data

null = df.isnull()
plt.figure(figsize=(10,6))
sns.heatmap(null, cmap = "viridis", cbar = False, yticklabels = False)
#plt.show()

unique_element = df.nunique()
print(f"The number of unique indices in each column:\n{unique_element}")

city_unique = df.city.nunique()
university_unique = df.university.nunique()

palet = ["#de9504", "#de5b04", "#de3e04", "#de3e04"]
input_features = ['country', 'program', 'level', 'duration_years', 'tuition_usd', 'living_cost_index', 'rent_usd', 'visa_fee_usd', 'insurance_usd', 'exchange_rate']
numerical_features = ['duration_years', 'tuition_usd', 'living_cost_index', 'rent_usd', 'visa_fee_usd', 'insurance_usd', 'exchange_rate']
categorical_features = ['country', 'program']

plt.figure(figsize=(16, 16))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(4,2,i)
    sns.histplot(df[feature], kde=True, color="#de9504")
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,6))
sns.set(font_scale=0.7)
for i, feature in enumerate(categorical_features, 1):
    print(df[feature].value_counts().index)
    print(df[feature].value_counts())
    plt.subplot(1,2,i)
    sns.countplot(df, y=feature, hue=feature, stat="count")
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

fig = px.scatter(
    df,
    x = "country",
    y = "tuition_usd",
    color="country",
    hover_data=["program", "level"]
)

fig.show()

#Removing the columns with lots of duplicate values
df.drop('level', axis='columns', inplace=True)
df.drop('duration_years', axis='columns', inplace=True)
df.drop('visa_fee_usd', axis='columns', inplace=True)
df.drop('insurance_usd', axis='columns', inplace=True)
print(f"DataFrame with cleaned columns:\n{df.head(5)}")



#Training the model
Data_train, data_temp = train_test_split(df, test_size = 0.3, random_state = 42)

#One-Hot Encoding for Categorical Data
encoder = OneHotEncoder(sparse_output=False)

#Encoding itself
categorical_data = Data_train.loc[:, Data_train.dtypes==object]
print(f"Categorical Data: \n {categorical_data}")
encoded_categorical_data = encoder.fit_transform(categorical_data)
feature_names = encoder.get_feature_names_out(categorical_data.columns)
df_encoded = pd.DataFrame(encoded_categorical_data, columns=feature_names)
print(f"Encoded Categorical Data: \n {df_encoded}")

#Dropping categorical columns
df.drop('country', axis='columns', inplace=True)
df.drop('city', axis='columns', inplace=True)
df.drop('university', axis='columns', inplace=True)
df.drop('program', axis='columns', inplace=True)

#Pasting encoded ones
merged_df = df.join(df_encoded)
merged_df.fillna(value = 0, axis="columns", inplace=True)
print(f"DataFrame with merged columns:\n {merged_df}")

Data_val, data_test = train_test_split(data_temp, test_size = 0.5, random_state = 30)

print(f"Training set: {Data_train.shape}")
print(f"Validation set: {Data_val.shape}")
print(f"Test set: {data_test.shape}")

#Merging sets
sets = {'Training Set': [Data_train], 'Validation Set': [Data_val], 'Test Set': [data_test]}
df_sets = pd.DataFrame(data=sets)
print(df_sets.head(10))

#Compute IQR to detect outliers
tuition_fee_mean = Data_train["tuition_usd"].mean()
tuition_fee_std = Data_train["tuition_usd"].std()
tuition_fee_count = Data_train["tuition_usd"].tolist()
print(f"Count: \n{tuition_fee_count}")
tuition_fee_median = Data_train["tuition_usd"].median()
Q_2_median = tuition_fee_median
Q_1_set = []
Q_3_set = []
print(f"Median: \n{tuition_fee_median}")
for i, x in enumerate(tuition_fee_count):
    if(x < tuition_fee_median):
       a = tuition_fee_count.pop(i)
       Q_1_set.append(a)
for k, x in enumerate(tuition_fee_count):
    if(x > tuition_fee_median):
        b = tuition_fee_count.pop(k)
        Q_3_set.append(b)

Q_1_set.sort()
Q_3_set.sort()
print(Q_1_set)
print(Q_3_set)
Q1_series = pd.Series(Q_1_set, copy=False)
Q3_series = pd.Series(Q_3_set, copy=False)
Q1 = Q1_series.median()
Q3 = Q3_series.median()
IQR = Q3 - Q1
print(f"Value of Q1: {Q1}")
print(f"Value of Q3: {Q3}")
print(f"Interquartile range: {IQR}")
outliers = Data_train[(Data_train["tuition_usd"] < (Q1 - 1.5*IQR)) | (Data_train["tuition_usd"] > (Q3 + 1.5*IQR))]
print(f"Outliers count: {len(outliers)}")

#StandardScaler

scaler = StandardScaler(with_mean=True, with_std=True)
scaled = scaler.fit_transform(merged_df)
scaled_df = pd.DataFrame(scaled, columns=merged_df.columns)
print(f"Scaled DataFrame: \n {scaled_df}")








