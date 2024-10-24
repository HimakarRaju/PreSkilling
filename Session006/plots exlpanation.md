Various Plots

1. **Bar Chart**: Cost by Aircraft Make/Model
2. **Histogram**: Distribution of Total Cost
3. **Box Plot**: Cost by Aircraft Make/Model
4. **Scatter Plot**: Wildlife Struck vs. Cost
5. **Count Plot**: Aircraft Types
6. **PairPlot**: Scatter Matrix of selected features
7. **Heat Map**: Correlation between numeric parameters

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset from Excel file
file_path = 'bsf.xlsx'
df = pd.read_excel(file_path)

# Convert 'Cost: Total $' to numeric
df['Cost: Total $'] = pd.to_numeric(df['Cost: Total $'], errors='coerce')

# Drop rows where 'Cost: Total $' is NaN
df_cleaned = df.dropna(subset=['Cost: Total $'])

# 1. Bar Chart: Cost by Aircraft Make/Model
plt.figure(figsize=(10, 6))
plt.bar(df_cleaned['Aircraft: Make/Model'], df_cleaned['Cost: Total $'], color='skyblue')
plt.xlabel('Aircraft: Make/Model')
plt.ylabel('Cost: Total $')
plt.title('Cost by Aircraft: Make/Model')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 2. Histogram: Distribution of Total Cost
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['Cost: Total $'], bins=20, color='purple')
plt.xlabel('Cost: Total $')
plt.ylabel('Frequency')
plt.title('Distribution of Total Cost')
plt.tight_layout()
plt.show()

# 3. Box Plot: Cost by Aircraft Make/Model
plt.figure(figsize=(12, 6))
sns.boxplot(x='Aircraft: Make/Model', y='Cost: Total $', data=df_cleaned)
plt.xticks(rotation=90)
plt.xlabel('Aircraft: Make/Model')
plt.ylabel('Cost: Total $')
plt.title('Cost by Aircraft Make/Model (Box Plot)')
plt.tight_layout()
plt.show()

# 4. Scatter Plot: Wildlife Struck vs. Cost
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned['Wildlife: Number struck'], df_cleaned['Cost: Total $'], alpha=0.7, color='green')
plt.xlabel('Wildlife Struck')
plt.ylabel('Cost: Total $')
plt.title('Wildlife Struck vs. Cost')
plt.tight_layout()
plt.show()

# 5. Count Plot: Aircraft Types
plt.figure(figsize=(10, 6))
sns.countplot(y='Aircraft: Type', data=df_cleaned, palette='coolwarm')
plt.title('Count of Aircraft Types')
plt.tight_layout()
plt.show()

# 6. PairPlot: Scatter Matrix (for selected columns)
sns.pairplot(df_cleaned[['Wildlife: Number struck', 'Cost: Total $', 'Feet above ground', 'Number of people injured']])
plt.show()

# 7. Heat Map: Correlation Between Numeric Parameters
plt.figure(figsize=(8, 6))
correlation = df_cleaned[['Wildlife: Number struck', 'Cost: Total $', 'Feet above ground', 'Number of people injured']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Numeric Parameters')
plt.tight_layout()
plt.show()
```

### Explanation of each plot

1. **Bar Chart**: Shows the total cost for each aircraft make/model. The x-axis has the aircraft model, and the y-axis shows the total cost.
2. **Histogram**: Plots the distribution of the total cost across all records. This shows how costs are spread out.
3. **Box Plot**: Displays the distribution of costs per aircraft make/model, indicating outliers and the spread of data.
4. **Scatter Plot**: Plots the number of wildlife strikes against the cost. Each point represents a record, allowing you to see if more strikes correspond to higher costs.
5. **Count Plot**: Visualizes the count of different aircraft types in the dataset.
6. **PairPlot**: A scatter matrix (pair plot) that shows pairwise relationships between selected numeric variables, helping to visualize interactions and distributions.
7. **Heat Map**: Shows correlations between numeric parameters, providing insight into relationships between features.
