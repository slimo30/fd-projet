import pandas as pd

df_orig = pd.read_csv(r'C:\Users\akaro\Documents\GitHub\fd-projet1\DATA-mining-project\uploads\Drug_Consumption.csv')
df_test = pd.read_csv(r'C:\Users\akaro\Documents\GitHub\fd-projet1\DATA-mining-project\uploads\test.csv')

print(f"Original shape: {df_orig.shape}")
print(f"Test shape: {df_test.shape}")

# Check if they are identical
identical = df_orig.equals(df_test)
print(f"Datasets are identical: {identical}")

# Compare columns
diff_cols = set(df_orig.columns) ^ set(df_test.columns)
print(f"Different columns: {diff_cols}")

# Check first few drug values
drug_cols = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
print("\nFirst 5 rows of drug values in test.csv:")
print(df_test[drug_cols].head())

# Check types
print("\nColumn types in test.csv:")
print(df_test.dtypes.head(20))
