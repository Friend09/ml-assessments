{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Clustering Assessment (Key)\n",
    "\n",
    "## Objective\n",
    "Identify customer segments based on their purchasing behavior."
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
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.cluster import KMeans\n",
    "\n",
    "# Load the dataset\n",
    "df = pd.read_csv('customer_data.csv')\n",
    "\n",
    "# Check for missing values or outliers\n",
    "print(df.isnull().sum())\n",
    "print(df.describe())\n",
    "\n",
    "# No missing values in this dataset, but let's check for outliers\n",
    "plt.figure(figsize=(12, 8))\n",
    "df.boxplot()\n",
    "plt.title('Boxplot of Features')\n",
    "plt.show()\n",
    "\n",
    "# For this example, we'll assume no outlier removal is necessary"
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
    "# Display summary statistics of the features\n",
    "print(df.describe())\n",
    "\n",
    "# Create a scatter plot of annual income vs. spending score\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.scatterplot(data=df, x='annual_income', y='spending_score')\n",
    "plt.title('Annual Income vs. Spending Score')\n",
    "plt.show()\n",
    "\n",
    "# Plot histograms for each feature\n",
    "df.hist(figsize=(12, 8))\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Feature Scaling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Normalize the features using StandardScaler\n",
    "scaler = StandardScaler()\n",
    "df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)\n",
    "\n",
    "# Explanation\n",
    "'''\n",
    "Feature scaling is important for clustering algorithms because:\n",
    "1. It ensures all features contribute equally to the distance calculations.\n",
    "2. It prevents features with larger magnitudes from dominating the clustering process.\n",
    "3. It improves the convergence of clustering algorithms.\n",
    "4. It makes the interpretation of the results more straightforward.\n",
    "'''"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Clustering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Implement the elbow method to find the optimal number of clusters\n",
    "inertias = []\n",
    "for k in range(1, 11):\n",
    "    kmeans = KMeans(n_clusters=k, random_state=42)\n",
    "    kmeans.fit(df_scaled)\n",
    "    inertias.append(kmeans.inertia_)\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.plot(range(1, 11), inertias, marker='o')\n",
    "plt.xlabel('Number of clusters (k)')\n",
    "plt.ylabel('Inertia')\n",
    "plt.title('Elbow Method for Optimal k')\n",
    "plt.show()\n",
    "\n",
    "# Choose the number of clusters\n",
    "n_clusters = 4  # This is an example, the actual number may vary\n",
    "\n",
    "# Explanation\n",
    "'''\n",
    "Based on the elbow plot, we choose 4 clusters because:\n",
    "1. There's a clear \"elbow\" at this point.\n",
    "2. The reduction in inertia becomes less significant after this point.\n",
    "3. It provides a balance between the number of clusters and the compactness of clusters.\n",
    "'''\n",
    "\n",
    "# Perform K-means clustering\n",
    "kmeans = KMeans(n_clusters=n_clusters, random_state=42)\n",
    "df['Cluster'] = kmeans.fit_predict(df_scaled)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Visualization and Interpretation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a scatter plot of the clustered data\n",
    "plt.figure(figsize=(12, 8))\n",
    "sns.scatterplot(data=df, x='annual_income', y='spending_score', hue='Cluster', palette='deep')\n",
    "plt.title('Customer Segments')\n",
    "plt.show()\n",
    "\n",
    "# Calculate and display the cluster centroids\n",
    "centroids = df.groupby('Cluster').mean()\n",
    "print(\"Cluster Centroids:\")\n",
    "print(centroids)\n",
    "\n",
    "# Interpretation\n",
    "'''\n",
    "Interpretation of the clusters:\n",
    "1. Cluster 0: Young, low income, high spenders (potential students or young professionals)\n",
    "2. Cluster 1: Middle-aged, high income, high spenders (affluent professionals)\n",
    "3. Cluster 2: Older, medium income, low spenders (conservative spenders, possibly retirees)\n",
    "4. Cluster 3: Middle-aged, medium income, medium spenders (average customers)\n",
    "\n",
    "These segments can be used for targeted marketing strategies, product recommendations, \n",
    "or customer retention programs tailored to each group's characteristics.\n",
    "'''"
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
