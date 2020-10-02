# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 00:36:47 2019

@author: Arsh Modak
"""


#%%

# Importing the libraries Numpy, Pandas, MatplotLib and Seaborn

import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)

#%%

# Importing the DataSet from its location;
# The delimiter used is "\t" (tab)
# All NA values are changed to the default missing value indicator for Python: NAN;
# Index column = 0 means the Index column is excluded in the DataSet

loan_credit = pd.read_csv(r'', delimiter = '\t', na_values = 'NaN', index_col = 0)
loan_credit.head()

#%%

# To Check the Data Types of the Variables

loan_credit.dtypes


#%%

# To check the shape (Rows, Columns) of the DataSet

loan_credit.shape

#%%

# To check the SUM of NULL Values in each variable

loan_credit.isnull().sum()

#%%

# Creating new variable and populating it with our DataSet 

loan_credit1 = loan_credit

#%%

# To Set the threshold for the deletion of columns containing Null Values

half_count= round(len(loan_credit1)/2, 0)

print(half_count)

#%%

# Dropping the Variables that contain Null Values above the set threshold value

loan_credit1 = loan_credit1.dropna(thresh = half_count, axis = 1)
print(loan_credit1.isnull().sum())

#%%

# To Check the Statistical Descripton of the DataSet

loan_credit.describe(include = 'all')


#%%

# Checking the Column Names of the DataSet:

loan_credit1.columns

#%%
            
# Creating a list of columns you want to drop from the DataSet.

columns_to_drop = ['sub_grade','title','zip_code',
           'addr_state','earliest_cr_line','last_pymnt_d','next_pymnt_d',
           'last_credit_pull_d', 'collections_12_mths_ex_med']


#%%

# Dropping the Columns

loan_credit1.drop(columns_to_drop, axis = 1, inplace = True )

#%%

loan_credit1.isnull().sum()

#%%

# Replacing the Values in "emp_length" from categorical to numerical.
# This is known as Manual Label Encoding

loan_credit1['emp_length'] = loan_credit1['emp_length'].replace({'2 years': 2, '1 year': 1, '4 years': 4, '8 years': 8,
                                                                 '10+ years': 10, '9 years': 9.0, '< 1 year': 0, '6 years': 6,
                                                                 '7 years': 7, '3 years': 3, '5 years': 5})


#%%

loan_credit1.shape


#%%

# Treating Missing Values:

# For Continous Variables we use the mean to impute missing values:

colnames_int=['tot_coll_amt','tot_cur_bal','total_rev_hi_lim','revol_util']

for y in colnames_int:
    loan_credit1[y].fillna(loan_credit1[y].mean(),inplace=True)

            
#%%
    
# Finding the Mean Value of "Emp_Length"
    
print(round(loan_credit1['emp_length'].mean(),0))

#%%

# Imputing Missing Values in "Emp_Length" by its Mean

loan_credit1['emp_length'].fillna(round(loan_credit1['emp_length'].mean(), 0), inplace = True)

    
#%%

# For Categorical Variables we use Mode to impute missing values:

loan_credit1['emp_title'].fillna(loan_credit1['emp_title'].mode()[0], inplace = True)


#%%

# To check the total number of null values in the DataSet:
          
loan_credit1.isnull().any(axis = 1).sum()

#%%

#loan_credit1.to_csv(r'C:\Users\Arsh Modak\Desktop\IMARTICUS\Python\loan_credTableau.csv', index= True, header= True)

#%%

loan_credit1.shape
loan_credit1['term'].unique()

#%%


# Manual Encoding for TERM

loan_credit1['term'].unique()

#%%

loan_credit1['term'] = loan_credit1['term'].replace({' 36 months':0, ' 60 months':1})

loan_credit1['term'].unique()

#%%


# Manual Encoding for VERIFICATION STATUS

loan_credit1['verification_status'].unique()

#%%

loan_credit1['verification_status'] = loan_credit1['verification_status'].replace({'Not Verified':0, 'Verified':1, 'Source Verified': 2})

loan_credit1['verification_status'].unique()

#%%


# Manual Encoding for HOME OWNERSHIP

loan_credit1['home_ownership'].unique()

#%%

loan_credit1['home_ownership'] = loan_credit1['home_ownership'].replace({'MORTGAGE':0, 'OWN':1,'RENT':2, 'OTHER':3,
                                                                         'NONE':4, 'ANY':5 })
    
loan_credit1['home_ownership'].unique()

#%%

loan_credit1.issue_d.head(5)

#%%

# SPLITTING THE DATA INTO TRAIN AND TEST

#%%

loan_credit1['issue_d'] = pd.to_datetime(loan_credit1['issue_d'])
col_name = 'issue_d'

#%%

# print(loan_credit[col_name].dtype)

#%%

loan_credit1.issue_d.head(5)

#%%

train_df = loan_credit1[loan_credit1['issue_d'] <= '2015-05-01']
test_df = loan_credit1[loan_credit1['issue_d'] > '2015-05-01']

#%%

# print(train_df.shape)
# print(test_df.shape)

#%%

# DROPPING COLUMN : "Issue_D"

train_df1=train_df.drop('issue_d',axis=1)
test_df1=test_df.drop('issue_d',axis=1)

#%%

# print(train_df1.shape)
# print(test_df1.shape)

#%%

# 'emp_title', 'issue_id', 'pymnt_plan', 'purpose', 'initial_list_status', 'application_type'4

#%%

# Creating a List of Categorical Variables for Label Encoding:

columndataLE = ['grade', 'emp_title', 'purpose', 'application_type', 'pymnt_plan', 'initial_list_status']

#%%

# PERFORMING LABEL ENCODING ON CATEGORICAL VARIABLES
# THIS CONVERTS CATEGORICAL DATA INTO NUMERICAL DATA

colname3 = columndataLE

le={}    
for z in colname3:
    le[z]=preprocessing.LabelEncoder()                   #LabelEncoder() is assigning labels to the key 

for z in colname3:
    train_df1[z]=le[z].fit_transform(train_df1[z])
    test_df1[z]=le[z].fit_transform(test_df1[z])

#%%

# print(train_df1.head()) 
# print(test_df1.head())

#%%

train_df.columns

#%%

test_df1['default_ind'].value_counts()

#%%

# CREATING X AND Y ARRAYS FOR TRAINING AND TESTING

X_train=train_df1.values[:, :-1] # Contains all rows and all columns besides "default_ind"
Y_train=train_df1.values[:,-1] # Contains all rows and only "default_ind"

X_test=test_df1.values[:, : -1] 
Y_test=test_df1.values[:, -1]

#%%

# STANDARDIZATION OF THE DATA SET

scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train) 
 
X_test = scaler.transform(X_test) 

# print(X_train)    
# print(X_test)


#%%

# train_df1.head()

#%%

# train_df1.dtypes

#%%

# Function to print metrics:
    
def printMetrics(Y_test, Y_pred):
    cfm = confusion_matrix(Y_test, Y_predGB)
    print("Confusion Matrix: ")
    print()
    print(cfm)
    
    print()
    print()
    print("Classification Report:" )
    print()
    print(classification_report(Y_test, Y_predGB))
    
    print()
    
    acc = accuracy_score(Y_test, Y_predGB)
    print("Accuracy of the Model: ", acc)  
    
    return


#%%

# USING LOGISTIC REGRESSION

# Create a model
classifier = LogisticRegression()

# Fitting Training data into the model
classifier.fit(X_train, Y_train) # fit is used to train the data  classifier.fit(dependent, independent)

Y_pred = classifier.predict(X_test)


# print(list(zip(Y_test, Y_pred)))

printMetrics(Y_test, Y_pred) # 99.87


#%%

# NAIVE BAYES

# alpha is the smoothing parameter (used to handle laplacian correction estimate)

modelNB = BernoulliNB(alpha = 1.0)
modelNB.fit(X_train, Y_train)

Y_pred_NB = modelNB.predict(X_test)

printMetrics(Y_pred, Y_predNB) #99.86

#%%

 # USING CROSS VALIDATION: LOGISTIC REGRESSION
 
#from sklearn.svm import SVC 

classifier = LogisticRegression()

#classifier = svm.SVC(kernel = 'rbf', C = 1, gamma = 0.001)

# performing kfold_cross_validation
kfold_cv = cross_validation.KFold(n = len(X_train), n_folds = 20)
print(kfold_cv)

#Running the model using scoring metric as Accuracy

kfold_cv_result = cross_validation.cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = kfold_cv)

#print(kfold_cv_result)

#finding the mean
print(kfold_cv_result.mean()) # 99.659

# MAX ACCURACY OF KFOLD FOR LOGISTIC REGRESSION:

print(max(kfold_cv_result)) # 99.81

#%%

# RUNNING EXTRATREES CLASSIFIER

modelETC = ExtraTreesClassifier(50, random_state = 10) # by default (10, random_state = 10)

# fit the model on the data and predict the values

modelETC = modelETC.fit(X_train, Y_train)
Y_predETC = modelETC.predict(X_test)

printMetrics(Y_pred, Y_predETC) # 59.123

#%%

# RUNNING RANDOMFOREST CLASSIFIER

modelRFC = RandomForestClassifier(501, random_state = 10) # by default (10, random_state = 10)

# fit the model on the data and predict the values

modelRFC = modelRFC.fit(X_train, Y_train)
Y_predRFC = modelRFC.predict(X_test)

printMetrics(Y_pred, Y_predRFC)# 99.838

#%%

# Predicting Using the Decision Tree Classifier:

model_DT = DecisionTreeClassifier(criterion = 'gini', random_state = 10)

# Fit the model on the data and predict the values

model_DT.fit(X_train, Y_train)
Y_predDT = model_DT.predict(X_test)

printMetrics(Y_pred, Y_predDT) #99.68 for both gini and entropy

#%%

            # BOOSTING

# Predicting using AdaBoost_Classifier

# base_estimator is to specify the classifier, n_estimator is the no. of models to be run.
model_AB = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(), n_estimators = 100, random_state = 10)
#model_AB = AdaBoostClassifier(base_estimator = svm.SVC(kernel = 'rbf', C= 1.0, gamma = 0.1), algorithm = 'SAMME', n_estimators = 100, random_state = 10)
#model_AB = AdaBoostClassifier(base_estimator = LogisticRegression(), n_estimators = 100, random_state = 10)
# Fit the model on the data and predict the values

model_AB.fit(X_train, Y_train)

Y_predAB = model_AB.predict(X_test)


#%%

# METRICS FOR ADABOOSTCLASSIFIER

printMetrics(Y_pred, Y_predAB) # 99.60

#%%

# Predicting Using the Gradient Boosting Classifier

model_GB = GradientBoostingClassifier(random_state = 10)

# fit the model on the data and predict values:

model_GB.fit(X_train, Y_train)

Y_predGB = model_GB.predict(X_test) 

#%%

# METRICS FOR GRADIENTBOOSTINGCLASSIFIER
printMetrics(Y_test, Y_predGB)

#%%



    





