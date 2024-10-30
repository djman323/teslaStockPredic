import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv(r'f:/ai/microProjectIML/Tesla.csv')

# Streamlit interface
st.title('Tesla Stock Price Analysis')

# Display the first few rows of the dataframe
st.write("### Tesla Stock Data", df.head())

# Plot 1: Tesla Close Price
st.write("### Tesla Close Price")
fig1, ax1 = plt.subplots(figsize=(15, 5))
ax1.plot(df['Close'])
ax1.set_title('Tesla Close Price', fontsize=15)
ax1.set_ylabel('Price in dollars')
st.pyplot(fig1)

# Plot 2: Distributions of Features
st.write("### Feature Distributions")
features = ['Open', 'High', 'Low', 'Close', 'Volume']

fig2, axes = plt.subplots(2, 3, figsize=(20, 10))

for i, col in enumerate(features):
    ax = axes[i // 3, i % 3]  # This handles the 2x3 grid
    sb.distplot(df[col], ax=ax)
    ax.set_title(f'{col} Distribution')

st.pyplot(fig2)

# Drop 'Date' column and group by year (if 
# 'year' column exists)
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    data_grouped = df.drop('Date', axis=1).groupby('year').mean()

    # Plot 3: Bar plots for Open, High, Low, Close over the years
    st.write("### Average Open, High, Low, Close by Year")
    fig3, axes3 = plt.subplots(2, 2, figsize=(20, 10))
    for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
        ax = axes3[i // 2, i % 2]
        data_grouped[col].plot(kind='bar', ax=ax)
        ax.set_title(f'Average {col} Price per Year')

    st.pyplot(fig3)
else:
    st.write("No 'Date' column found in the dataset.")
