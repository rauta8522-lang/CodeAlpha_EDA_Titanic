import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- TASK 2: EXPLORATORY DATA ANALYSIS (EDA) ---

# 1. Dataset Loading
# Using the Titanic dataset, which is ideal for performing EDA
try:
    df = sns.load_dataset('titanic')
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("Please check your internet connection to load seaborn datasets.")

# 2. Understanding Data Structure (Basic Exploration)
print("\n--- Dataset Overview (First 5 Rows) ---")
print(df.head()) 

# 3. Data Cleaning: Identifying Missing Values
print("\n--- Missing Values Analysis ---")
missing_data = df.isnull().sum()
# Display only columns that have missing values
print(missing_data[missing_data > 0])

# 4. Data Visualization (Patterns & Trends)
# Creating visualizations to understand survival rates and age distribution
plt.figure(figsize=(12, 5))

# Subplot 1: Survival Count by Gender
plt.subplot(1, 2, 1)
sns.countplot(data=df, x='survived', hue='sex', palette='viridis')
plt.title('Survival Count by Gender')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Passenger Count')

# Subplot 2: Age Distribution (Checking for density and outliers)
plt.subplot(1, 2, 2)
sns.histplot(df['age'].dropna(), kde=True, color='teal')
plt.title('Passenger Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 5. Correlation Heatmap (Feature Relationships)
# Analyzing how numerical features are related to each other
plt.figure(figsize=(8, 6))
# Selecting only numerical columns for correlation calculation
numeric_df = df.select_dtypes(include=['number']) 
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Features')
plt.show()

print("\n--- Final Task Summary ---")
print("1. Data Structure explored and verified.")
print("2. Missing values detected for cleaning.")
print("3. Survival patterns and age distribution visualized.")
print("✅ EDA Task 2 Completed Successfully!")