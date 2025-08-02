import pandas as pd

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