{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Regression Analysis Assessment (Key)\n",
    "\n",
    "## Objective\n",
    "Predict house prices based on various features."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Data Loading and Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import necessary libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "from sklearn.compose import ColumnTransformer\n",
    "\n",
    "# Load the dataset\n",
    "df = pd.read_csv('house_prices.csv')\n",
    "\n",
    "# Handle missing values (if any)\n",
    "df = df.dropna()\n",
    "\n",
    "# Encode categorical variables\n",
    "categorical_features = ['location']\n",
    "numeric_features = ['size_sqft', 'bedrooms', 'bathrooms', 'age_years', 'garage_spaces']\n",
    "\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('num', 'passthrough', numeric_features),\n",
    "        ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)\n",
    "    ])\n",
    "\n",
    "X = df.drop('price', axis=1)\n",
    "y = df['price']\n",
    "\n",
    "X_encoded = preprocessor.fit_transform(X)\n",
    "feature_names = numeric_features + preprocessor.named_transformers_['cat'].get_feature_names(categorical_features).tolist()\n",
    "X_encoded = pd.DataFrame(X_encoded, columns=feature_names)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Exploratory Data Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Display summary statistics of the numerical features\n",
    "print(df[numeric_features + ['price']].describe())\n",
    "\n",
    "# Create a correlation heatmap of the features\n",
    "plt.figure(figsize=(10, 8))\n",
    "sns.heatmap(X_encoded.corr(), annot=True, cmap='coolwarm')\n",
    "plt.title('Correlation Heatmap of Features')\n",
    "plt.show()\n",
    "\n",
    "# Plot a scatter plot of house size vs. price\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.scatterplot(data=df, x='size_sqft', y='price')\n",
    "plt.title('House Size vs. Price')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Feature Selection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Identify the top 5 features with strongest correlation to house price\n",
    "correlations = X_encoded.corrwith(y).abs().sort_values(ascending=False)\n",
    "top_5_features = correlations.head(5)\n",
    "print(\"Top 5 features correlated with price:\")\n",
    "print(top_5_features)\n",
    "\n",
    "# Explanation\n",
    "'''\n",
    "I chose these top 5 features because they have the strongest correlation with the house price. \n",
    "This indicates that they are likely to be the most influential in determining the price. \n",
    "However, it's important to note that correlation doesn't imply causation, and we should be \n",
    "cautious about multicollinearity among these features.\n",
    "'''"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Model Selection and Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestRegressor\n",
    "\n",
    "# Split the data into training and testing sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Choose and train the model\n",
    "model = RandomForestRegressor(n_estimators=100, random_state=42)\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Explanation\n",
    "'''\n",
    "I chose Random Forest Regressor because:\n",
    "1. It can capture non-linear relationships between features and the target variable.\n",
    "2. It's less prone to overfitting compared to a single decision tree.\n",
    "3. It can handle both numerical and categorical features well.\n",
    "4. It provides feature importance, which can give insights into which features are most predictive.\n",
    "5. It often performs well out-of-the-box without extensive hyperparameter tuning.\n",
    "'''"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Making Predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import mean_squared_error, r2_score\n",
    "\n",
    "# Make predictions on the test set\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "# Calculate and display the Mean Squared Error (MSE) and R-squared score\n",
    "mse = mean_squared_error(y_test, y_pred)\n",
    "r2 = r2_score(y_test, y_pred)\n",
    "print(f'Mean Squared Error: {mse:.2f}')\n",
    "print(f'R-squared Score: {r2:.2f}')\n",
    "\n",
    "# Plot the predicted prices against the actual prices\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.scatter(y_test, y_pred, alpha=0.5)\n",
    "plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)\n",
    "plt.xlabel('Actual Price')\n",
    "plt.ylabel('Predicted Price')\n",
    "plt.title('Actual vs Predicted House Prices')\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
