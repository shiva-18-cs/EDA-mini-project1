import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
import numpy as np

import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('data1.csv')

#EDA

print(df.head())

print(df.info())
print(df.describe())

print(df.isnull().sum())

print(df.columns)

#visulaization

numeric_cols=['age', 'bmi', 'children']

for col in numeric_cols:
    plt.figure(figsize=(8,4))
    sns.histplot(df[col], kde=True)
    plt.show()
    
numeric_cols = ['age', 'bmi', 'children']

plt.figure(figsize=(15, 4))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(1, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.show()
    

sns.countplot(x=df['children'])    
plt.show()
sns.countplot(x=df['smoker'])
plt.show()

for col in numeric_cols:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=df[col])
    plt.show()
plt.figure(figsize=(15, 4))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10,6))    
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

#data cleaning and preprocessing
'''here we create a copy of the original dataframe to perform data cleaning and preprocessing 
without altering the original data.'''
print("copie of original dataframe")
df_cleaned=df.copy()
print(df_cleaned.shape)
df_cleaned.drop_duplicates(inplace=True)
print("after removing duplicates")
print(df_cleaned.shape)
df_cleaned.isnull().sum()

print(df_cleaned.dtypes)#checking data types of each column
'''when we get object data type it means string like data .we need to convert them into numerical format 
for that we use one hot encoding and label encoding techniques.for sex and smoker columns we use label encoding.for
region column we use one hot encoding technique.'''

df_cleaned['sex'].value_counts()
df_cleaned['sex']=df_cleaned['sex'].map({"male":0,"female":1})

df_cleaned['smoker'].value_counts()
df_cleaned['smoker']=df_cleaned['smoker'].map({"yes":1,"no":0})

df_cleaned.rename(columns={'smoker':'is_smoker',
                           'sex':'is_female'},inplace=True)
# Convert the 'region' categorical column into dummy/one-hot encoded columns.
# drop_first=True removes the first dummy column to avoid multicollinearity
# (dummy variable trap), keeping only n-1 columns.
'''Machine learning models cannot understand text (like “southwest”, “northwest”).
So we convert each category into numbers (0 or 1).'''
df_cleaned=pd.get_dummies(df_cleaned,columns=['region'],drop_first=True)

df_cleaned.head()
df_cleaned.astype(int)
print(df_cleaned.head())

#feature engineering 
'''Feature engineering is the process of using domain knowledge to extract features from raw data via
various techniques. It helps to improve the performance of machine learning models.'''

df_cleaned['bmi_category'] = pd.cut(df_cleaned['bmi'], bins=[0, 18.5, 25, 30, 100], 

                                        labels=['Underweight', 'Normal weight', 'Overweight', 'Obesity'])

df_cleaned=pd.get_dummies(df_cleaned,columns=['bmi_category'],drop_first=True)
df.cleaned['bmi_category'].astype(int)
print(df_cleaned.head())

#scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
cols=['age','bmi','children']
df_cleaned[cols]=scaler.fit_transform(df_cleaned[cols])
print(df_cleaned.head())

#feature extraction
'''Feature extraction is the process of transforming raw data into a set of features that can be used'''

from scipy.stats import pearsonr

#pearson correlation 

#list of features to check against target variable 'charges'
df_cleaned.columns

# Pearson Correlation Calculation
# --------------------------------

# List of features to check against target
selected_features = [
    'age', 'bmi', 'children', 'is_female', 'is_smoker',
    'region_northwest', 'region_southeast', 'region_southwest',
    'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese'
]

correlations = {
    feature: pearsonr(df_cleaned[feature], df_cleaned['charges'])[0]
    for feature in selected_features
}

correlation_df = pd.DataFrame(
    list(correlations.items()),
    columns=['Feature', 'Pearson Correlation']
)
correlation_df.sort_values(by='Pearson Correlation', ascending=False, inplace=True)

print(correlation_df)
#chi square test for categorical features
cat_features = [
    'is_female', 'is_smoker',
    'region_northwest', 'region_southeast', 'region_southwest',
    'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese'
]

from scipy.stats import chi2_contingency
import pandas as pd

alpha = 0.05

# Bin target variable (charges) into quartiles
df_cleaned['charges_bin'] = pd.qcut(
    df_cleaned['charges'],
    q=4,
    labels=False
)

chi2_results = {}

for col in cat_features:
    contingency = pd.crosstab(df_cleaned[col], df_cleaned['charges_bin'])
    chi2_stat, p_val, _, _ = chi2_contingency(contingency)

    decision = (
        'Reject Null (Keep Feature)'
        if p_val < alpha
        else 'Accept Null (Drop Feature)'
    )

    chi2_results[col] = {
        'chi2_statistic': chi2_stat,
        'p_value': p_val,
        'Decision': decision
    }

# Convert results to DataFrame and sort by p-value
chi2_df = pd.DataFrame(chi2_results).T
chi2_df = chi2_df.sort_values(by='p_value')

chi2_df
print(chi2_df)

