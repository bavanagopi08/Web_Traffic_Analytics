import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create output folder
os.makedirs("output", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset/website_wata.csv")

print("WEB TRAFFIC ANALYTICS")
print("=" * 40)

# Basic information
print("\nDataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Fill missing values safely
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna("Unknown")

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv("output/cleaned_website_data.csv", index=False)

# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(include="object").columns

# Generate numeric summary
df[numeric_cols].describe().to_csv(
    "output/numeric_summary.csv"
)

# Generate categorical summary
with open("output/traffic_report.txt", "w") as f:
    f.write("WEB TRAFFIC ANALYTICS REPORT\n")
    f.write("=" * 40 + "\n")
    f.write(f"Total Records: {len(df)}\n")
    f.write(f"Total Columns: {len(df.columns)}\n")
    f.write(f"Duplicate Records: {df.duplicated().sum()}\n")
    f.write("\nMissing Values:\n")
    f.write(df.isnull().sum().to_string())
    f.write("\n\nNumeric Summary:\n")
    f.write(df[numeric_cols].describe().to_string())
    f.write("\n\nCategorical Columns:\n")
    f.write(str(list(categorical_cols)))

# Create graphs for numeric columns
sns.set_theme(style="whitegrid")

for col in numeric_cols[:4]:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(f"output/{col}_distribution.png")
    plt.close()

# Correlation heatmap
if len(numeric_cols) >= 2:
    plt.figure(figsize=(9, 6))
    sns.heatmap(
        df[numeric_cols].corr(),
        annot=True,
        cmap="coolwarm"
    )
    plt.title("Website Traffic Correlation")
    plt.tight_layout()
    plt.savefig("output/correlation_heatmap.png")
    plt.close()

print("\nAnalysis completed successfully!")
print("Check the output folder.")