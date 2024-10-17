# Text Classification Assessment

## Objective
Classify customer reviews into positive or negative sentiment.

## Dataset
The dataset `customer_reviews.csv` contains customer reviews and their corresponding ratings (1-5 stars).

## Tasks

1. Data Loading and Preprocessing
   - Load the `customer_reviews.csv` file using pandas
   - Create a binary target variable: reviews with 4-5 stars are positive (1), others are negative (0)
   - Split the data into training and testing sets (80/20 split)

2. Exploratory Data Analysis
   - Display the first few rows of the dataset
   - Show the distribution of positive and negative reviews
   - Calculate and display the average length of reviews for each class

3. Feature Engineering
   - Use CountVectorizer or TfidfVectorizer to convert text data into numerical features
   - Explain your choice between CountVectorizer and TfidfVectorizer

4. Model Selection and Training
   - Choose an appropriate classification algorithm (e.g., Logistic Regression, Naive Bayes, or Random Forest)
   - Train the model on the training data
   - Explain your choice of algorithm

5. Making Predictions
   - Use the trained model to make predictions on the test set
   - Calculate and display the accuracy score of your model
   - Print a few example predictions along with their actual labels

## Submission
Submit your completed Jupyter notebook with all the code and explanations for each step.
