import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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




# --- Step 5: Build K-Means Model ---

# --- Find the optimal number of clusters using the Elbow Method ---
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# --- Build the final model with the optimal k ---
# Based on the elbow plot, let's choose 4 clusters.
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)

# Assign clusters to each customer
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# --- Analyze the clusters ---
print("\n--- Cluster Analysis ---")
# Calculate the average RFM values for each cluster
cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
print("Average RFM values for each cluster:")
print(cluster_summary)


# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=rfm, x='Recency', y='Frequency', hue='Cluster', palette='viridis', s=100, alpha=0.7)
plt.title('Customer Segments by Recency and Frequency')
plt.show()


# Create a mapping from cluster number to a descriptive name
# NOTE: The cluster numbers and their meanings might change each time you run the model
# You must look at the `cluster_summary` table above to define this mapping correctly.
# For this example, let's assume the following profile based on a typical run:
# Cluster 0: High F, High M, Low R -> Champions
# Cluster 1: Low F, Low M, High R -> At-Risk
# Cluster 2: Low F, Low M, Low R  -> New Customers
# Cluster 3: Mid F, Mid M, Mid R  -> Potential Loyalists

cluster_map = {
    0: 'Champions 🏆',
    1: 'At-Risk 😟',
    2: 'New Customers 🌱',
    3: 'Potential Loyalists ⭐'
}

# Add a new column with the descriptive label
rfm['Segment'] = rfm['Cluster'].map(cluster_map)


print("\n--- Final Segments ---")
# Display the first 10 customers with their new segment label
print(rfm.head(10))

# Visualize the segments with new labels
plt.figure(figsize=(10, 6))
sns.scatterplot(data=rfm, x='Recency', y='Frequency', hue='Segment', palette='viridis', s=100, alpha=0.7)
plt.title('Customer Segments by Recency and Frequency')
plt.show()