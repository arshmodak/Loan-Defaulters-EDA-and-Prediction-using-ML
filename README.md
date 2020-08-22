# Loan-Defaulters-EDA-and-Prediction-using-ML

The dataset consists of over 850k observations and 72 features. (Refer Data Dictionary for features)

The task here was to predict if an individual would pay back their loans (non-defaulter) or not (defaulter).

The data was thoroughly preprocessed:
Columns with over 50% missing values and if redundant were removed, else imputed using either the mode or median. (Future works with knn imputation)
Manual and library-based label encoding was performed to make the data easier to read for the ML models.

The data was standardized and then used for Machine Learning. The data was also used to derive insights. (Mentioned in Documentation).

The Machine Learning Algorithms used to create the models are as follows:

1. Logistic Regression
2. Decision Trees
3. Random Forest
4. Naive Bayes (Gaussian and Bernoulli)
5. AdaBoost
6. Gradient Boosting

The models were improved using K-Fold Cross Validation and were evaluated using metrics such as Recall, Precision, F1-Score and Accuracy
