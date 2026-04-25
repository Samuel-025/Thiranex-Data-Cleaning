import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- PHASE 1: DATA LOADING ---
print("Loading dataset...")
# Using the Titanic dataset as it's perfect for a "Raw Data" task
df = sns.load_dataset('titanic')

print(f"Initial Data Shape: {df.shape}")
print("\nFirst 5 rows of raw data:")
print(df.head())

# --- PHASE 2: DATA CLEANING ---
print("\n--- Starting Data Cleaning ---")

# 1. Handling Missing Values
print("Missing values before cleaning:")
print(df.isnull().sum())

# Fill 'age' with the median (best practice for skewed distributions)
df['age'] = df['age'].fillna(df['age'].median())

# Fill 'embarked' with the most common value (mode)
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Drop 'deck' column because >75% of it is missing (unusable)
if 'deck' in df.columns:
    df.drop(columns=['deck'], inplace=True)

# 2. Removing Duplicates
duplicate_count = df.duplicated().sum()
print(f"Removing {duplicate_count} duplicate rows...")
df.drop_duplicates(inplace=True)

# 3. Handling Outliers in 'Fare'
# We'll cap fares at 300 to ensure visualizations aren't squashed by a few extreme values
print("Filtering outliers in 'fare' column...")
df = df[df['fare'] < 300]

print("\nCleaning Complete. Final Data Shape:", df.shape)

# --- PHASE 3: VISUALIZATION & INSIGHTS ---
print("\nGenerating Visualizations...")

# Insight 1: Survival Probability by Class and Gender
# This shows the "Story" of the data clearly
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='class', y='survived', hue='sex', palette='magma')
plt.title('Survival Rate by Passenger Class and Gender')
plt.ylabel('Survival Probability')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Insight 2: Age Distribution of Passengers
# We use the explicit 'data' and 'x' parameters to avoid Pylance type-checking errors
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', bins=25, kde=True, color='teal')
plt.title('Distribution of Passenger Ages (Cleaned)')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.show()

print("\nProject execution finished. You can now take screenshots of the charts for your Thiranex submission!")