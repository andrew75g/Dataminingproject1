import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# Simulate NBA player data
np.random.seed(0)  # For reproducibility

# Create a DataFrame with 'Age', 'Years_of_Experience', 'Player_Efficiency_Rating (PER)', 'Win_Shares (WS)', 'Box_Plus_Minus (BPM)'
player_data = pd.DataFrame({
    'Age': np.random.choice(range(19, 40), 500, replace=True),
    'Years_of_Experience': np.random.choice(range(1, 21), 500, replace=True),
    'PER': np.random.normal(15, 5, 500),  # PER is often around 15, with some variation
    'WS': np.random.normal(5, 2.5, 500),  # Win Shares can vary widely
    'BPM': np.random.normal(0, 6, 500)    # Box Plus/Minus can be both positive and negative
})

# Clean the data if necessary (here we assume data is already clean)

# EDA: Visualize the distribution of ages
plt.figure(figsize=(8, 6))
sns.histplot(player_data['Age'], kde=False, bins=range(19, 41))
plt.title('Distribution of Player Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Visualize the relationship between 'Age' and 'PER'
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='PER', data=player_data)
plt.title('Player Efficiency Rating by Age')
plt.xlabel('Age')
plt.ylabel('Player Efficiency Rating (PER)')
plt.show()

# Calculate and visualize the correlation matrix
correlation_matrix = player_data[['Age', 'Years_of_Experience', 'PER', 'WS', 'BPM']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Player Metrics')
plt.show()

# Simple Linear Regression: Effect of Age on PER
X = player_data['Age']
y = player_data['PER']
X = sm.add_constant(X)  # Adds a constant term to the predictor
model = sm.OLS(y, X).fit()
print(model.summary())


