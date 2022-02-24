###Bagging and Random Forest###

##Bagging##
#Definition:Bootstrap(引导程序) Aggregation,reduce the variance of individual models in the ensemble.Sample drawn from original set with replacement,any individual element can be drawn any times.

#Classification:BaddingClassifier(),aggregates predictions by matority voting.
#Import models and utility functions
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Set seed for reproducibility
SEED = 1
# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    stratify=y,
                                                    random_state=SEED)
#Instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16, random_state=SEED)
#Instantiate a BaggingClassifier 'bc'
bc = BaggingClassifier(base_estimator=dt, n_estimators=300, n_jobs=-1) #consists of 300 classification trees dt,n_job=-1 means all CPU cores are used in computation
#Fit 'bc' to the training set
bc.fit(X_train, y_train)
#Predict test set labels
y_pred = bc.predict(X_test)
#Evaluate and print test-set accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of Bagging Classifier: {:.3f}'.format(accuracy))

#Regression:using BaggingRegressor(),aggregate predictions through averaging.the examples are similar to the case above.


##Out of Bag Evaluation##(OOB)
#other instances may not be sampled at all,as usual some instances may be sampled several times for one model
#OOB Definition:on average, for each model,63% of the training instances are sampled;Then the remaining 37% is OOB instances,they are unseen when training the model,since they can be used to estimate the performance of ensemble without the need for CV-->OOB valuation.
#We label OOB samples for evaluation from OOB1 to OOBn；
#The OOB-Score of bagging ensemble is evaulated as the average of these N OOB scores, that is:OOB Score=(OOB1+OOB2+...+OOBn)/N

# Import models and split utility function
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Set seed for reproducibility
SEED = 1
# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size= 0.3,
                                                     stratify= y,
                                                     random_state=SEED)
# Instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=4,
                            min_samples_leaf=0.16,
                            random_state=SEED)
# Instantiate a BaggingClassifier 'bc'; set oob_score = True
bc = BaggingClassifier(base_estimator=dt, n_estimators=300,
                       oob_score=True, n_jobs=-1)
# Fit 'bc' to the training set
bc.fit(X_train, y_train)
# Predict the test set labels
y_pred = bc.predict(X_test)
# Evaluate test set accuracy
test_accuracy = accuracy_score(y_test, y_pred)
# Extract the OOB accuracy from 'bc'
oob_accuracy = bc.oob_score_
# Print test set accuracy
print('Test set accuracy: {:.3f}'.format(test_accuracy))
test set accuracy: 0.936
# Print OOB accuracy
print('OOB accuracy: {:.3f}'.format(oob_accuracy))
OOB accuracy: 0.925  #These 2 accuarcies are so close but not equal.OOB-Score do not need CV process.

 
##Random Forest##(RF)

#basic estimator:Decision tree
#Each estimator is trained on a different bootstrap sample with same size as the training set（bootstrap samples size= # of samples)
#d features are sampled at each node without replacement(d<total # of features).For sklearn d default to the (√# of features)
#The node is then split using the sampled feature that maximizes the information gain.

#RF for Classification:RandomForestClassifier() in sklearn
#RF for Regression:RandomForestRegressor() in sklearn  #in general,RF achieves a lower variance than individual trees

# Basic imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
# Set seed for reproducibility
SEED = 1
# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=SEED)
# Instantiate a random forests regressor 'rf' 400 estimators
rf = RandomForestRegressor(n_estimators=400,min_samples_leaf=0.12,random_state=SEED)
# Fit 'rf' to the training set
rf.fit(X_train, y_train)
# Predict the test set labels 'y_pred'
y_pred = rf.predict(X_test)
# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2)
# Print the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))
Test set RMSE of rf: 3.98


##Feature importance##--Find the most predictive features
#Tree-based method:enable measuring the importance of each feature in prediction. Use feature_importance_ in sklearn
#how much the tree nodes use a particular feature(weighted avg to illustrate as percentage) to reduce impurity

#Feature importance in sklearn
import pandas as pd
import matplotlib.pyplot as plt
# Create a pd.Series of features importances
importances_rf = pd.Series(rf.feature_importances_, index = X.columns) 
# Sort importances_rf
sorted_importances_rf = importances_rf.sort_values()
# Make a horizontal bar plot
sorted_importances_rf.plot(kind='barh', color='lightgreen'); plt.show()
