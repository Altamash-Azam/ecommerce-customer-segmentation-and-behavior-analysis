import pandas as pd
import datetime as dt

# Define the path to your dataset
file_path = 'Online Retail.xlsx'

# Load the data from the Excel file
try:
    df = pd.read_excel(file_path)
    print("✅ Data loaded successfully!")
    print(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
except FileNotFoundError:
    print(f"❌ Error: The file '{file_path}' was not found. Please make sure it's in the correct directory.")

# --- Step 2: Data Cleaning & Preprocessing ---

# Check for missing values in each column
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Drop rows where CustomerID is missing, as we can't analyze customers without an ID
df.dropna(axis=0, subset=['CustomerID'], inplace=True)

# Convert CustomerID to an integer type
df['CustomerID'] = df['CustomerID'].astype(int)

print("\nMissing values after dropping rows with null CustomerID:")
print(df.isnull().sum())

# Remove duplicate transactions
df.drop_duplicates(inplace=True)

# Remove canceled orders (those with InvoiceNo starting with 'C')
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# Ensure 'Quantity' and 'UnitPrice' are positive
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

print("\n--- Data Cleaning Complete ---")
print(f"The cleaned dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
print("\nFirst 5 rows of the cleaned dataset:")
print(df.head())



# --- Step 3: Feature Engineering (RFM) ---

# Calculate TotalPrice for each transaction
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# --- Calculate Recency ---
# To calculate recency, we need a 'snapshot' date to measure from.
# This will be one day after the last transaction in the dataset.
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

# Group by customer and find their last purchase date
recency_df = df.groupby('CustomerID').agg({'InvoiceDate': 'max'}).reset_index()
recency_df.rename(columns={'InvoiceDate': 'LastPurchaseDate'}, inplace=True)

# Calculate the number of days since the last purchase
recency_df['Recency'] = (snapshot_date - recency_df['LastPurchaseDate']).dt.days

# --- Calculate Frequency ---
# Group by customer and count the number of unique invoices
frequency_df = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
frequency_df.rename(columns={'InvoiceNo': 'Frequency'}, inplace=True)

# --- Calculate Monetary ---
# Group by customer and sum their total purchases
monetary_df = df.groupby('CustomerID')['TotalPrice'].sum().reset_index()
monetary_df.rename(columns={'TotalPrice': 'Monetary'}, inplace=True)

# --- Merge RFM dataframes ---
# Merge Recency and Frequency data
rfm = pd.merge(recency_df[['CustomerID', 'Recency']], frequency_df, on='CustomerID')

# Merge the result with Monetary data
rfm = pd.merge(rfm, monetary_df, on='CustomerID')

print("\n--- RFM Features Created ---")
print("RFM DataFrame (first 5 rows):")
print(rfm.head())

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- Step 4: Preprocess Data for Modeling ---

# --- Visualize data distribution ---
print("\n--- Visualizing data distributions before scaling ---")
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
sns.histplot(rfm['Recency'], ax=axes[0], kde=True).set_title('Recency Distribution')
sns.histplot(rfm['Frequency'], ax=axes[1], kde=True).set_title('Frequency Distribution')
sns.histplot(rfm['Monetary'], ax=axes[2], kde=True).set_title('Monetary Distribution')
plt.tight_layout()
plt.show()

# --- Apply Log Transformation to reduce skewness ---
# We use np.log1p which is log(1+x) to handle potential zero values
rfm_log = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])

# --- Scale the data ---
# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the log-transformed data
scaled_data = scaler.fit_transform(rfm_log)

# Create a new DataFrame with the scaled data
rfm_scaled = pd.DataFrame(scaled_data, columns=rfm_log.columns)


print("\n--- Data Preprocessing Complete ---")
print("Scaled RFM data (first 5 rows):")
print(rfm_scaled.head())