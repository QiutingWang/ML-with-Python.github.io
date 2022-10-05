###optimize  classification and regression models using hyperparameter turning###

##diagnosing classifaction predictions--Confusion Matix## For binary classification
                     #Predicted Spam Email                     Predicted Real Email
#Actual:Spam Email    True Positive 阳性                        False Negative 假阴性
#Actual:Real Email    False Positive  假阳性                     True Negative 阴性

#Accuracy from the confusion matrix:=(TP+TN)/(TP+TN+FP+FN)
#Precision:=TP/(TP+FP), positive predictive value(PPV); High precision means that not many real emails predicted as spam
#Recall:=TP/(TP+FN), sensitivity,hit rate,true positive rate;Predicted most spam emails correctly
#F1-Score:=2*(precision*recall)/(precision+recall)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
knn = KNeighborsClassifier(n_neighbors=8)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4, random_state=42)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
#print the confusion matrix and accuracy/precision/recall/f1-score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


##Logistic Regression and ROC curve##--in classifiction problem,not regression problem!
#output is probability:if p>0.5,it will be labeled as 1; if p<0.5-->0; The default threshold is 0.5.
#if p=0,TP=FP, the slope of ROC curve =1

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
logreg = LogisticRegression() #set the regressor
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4, random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

#plotting the ROC curve:
from sklearn.metrics import roc_curve
y_pred_prob = logreg.predict_proba(X_test)[:,1]  #predicted probabilities
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob) #unpacked the 3 variables:false positive rate,true positive rate,thresholds
plt.plot([0, 1], [0, 1], 'k--') #the x and y range
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show();
#Apply the method predict probability to the model and pass it to the test data
logreg.predict_proba(X_test)[:,1] #we want the probablity that our log reg model before the threshold to predict the label
#the index1:the probailities of the predicted label being 1


##Area under the ROC curve(AUC)--the larger, the better##
from sklearn.metrics import roc_auc_score
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4, random_state=42)
logreg.fit(X_train, y_train)
y_pred_prob = logreg.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_pred_prob) #compute the auc score,first compute the predicted probabilities as above

#we can always compute auc using cross-validation method:
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(logreg, X, y, cv=5,
                                scoring='roc_auc')
print(cv_scores) #then print the auc list


##Hyperparameter tuning##
#linear regression:choosing parameters,Ridge/Lasso regression-alpha,KNN-n_neighbors/K

#Grid search cross-validation(CV):we select what it performs best--GridSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': np.arange(1, 50)} #we select the hyperparameter like a dictionary, 取值范围
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5) #enter the number of folds want to use
knn_cv.fit(X, y) #fit the data
knn_cv.best_params_ #get the best parameter we gain
knn_cv.best_score_ #compute the accuracy score

#Randomlized-CV:EG:
# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}
# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()
# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv =RandomizedSearchCV(tree,param_dist, cv=5)
# Fit it to the data
tree_cv.fit(X, y)
# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


##Hold-Out Set##--to absolutely certain about your model's ability to generalize to unseen data.
#Split data into training and hold-out set at the beginning,then perform grid search cross-valiadation on training set-->choose the best hyperparameter and the score calculation
#Note:if we want to calculate MSE:
mse = mean_squared_error(y_test, y_pred)