####Feature Selection2:selecting for model accuracy####

###Selecting for feature performance###
##Pre-processing data##
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
##creating logistic regression##
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr = LogisticRegression()
lr.fit(X_train_std, y_train)
X_test_std = scaler.transform(X_test)
y_pred = lr.predict(X_test_std)
print(accuracy_score(y_test, y_pred)) #return:0.99
print(lr.coef_)
#return: array([[-3.  ,  0.14,  7.46,  1.22,  0.87]])
print(dict(zip(X.columns, abs(lr.coef_[0])))) #use zip function to transform the output into a dictionary that shows which feature has the corresponding coefficient. We can compare the coefficients with each other
#return: feature with coefficients close to zero will contribute little to the end result
{'chestdepth': 3.0,
 'handlength': 0.14,
 'neckcircumference': 7.46,
 'shoulderlength': 1.22,
 'earlength': 0.87}
#then we can remove the relative low coefficient in the model.   
#in this case, we remove handlength feature:
X.drop('handlength', axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
lr.fit(scaler.fit_transform(X_train), y_train)
print(accuracy_score(y_test, lr.predict(scaler.transform(X_test))))
#return: 0.99-->unchanged

#recursive feature elimination#RFE:Manual delete the features with lowest accuracy one by one/Automatic RFE
from sklearn.feature_selection import RFE
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=2, verbose=1) #we desire to remain 2 features at final
rfe.fit(X_train_std, y_train)
#return:
Fitting estimator with 5 features.
Fitting estimator with 4 features.
Fitting estimator with 3 features.
#dropping a feature will affect other feature's coefficient

#inspecting the RFE results:
X.columns[rfe.support_] #contains true/false values to see which features were kept in the dataset
#return:Index(['chestdepth', 'neckcircumference'], dtype='object')
print(dict(zip(X.columns, rfe.ranking_))) #checking the RFE's ranking attribute,to see which features we will drop
#return:
{'chestdepth': 1, #value=1, means the feature will be kept in the dataset until the end
 'handlength': 4,  #the high values will be droped early on
 'neckcircumference': 1,
 'shoulderlength': 2,
 'earlength': 3}
print(accuracy_score(y_test, rfe.predict(X_test_std)))
#return:0.99 it remains the same as above, which means the features we delete have little impact or no impact on the models we predict

###Tree-Based Feature Selection###
#EG:Random Forest Classifier to predict gender with 93 features in the dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(accuracy_score(y_test, rf.predict(X_test)))
#return:0.99-->escape the curse of dimensionality and didn't overfit on many features in the training set

#calculate feature importance values for decide which feature we use near the root, which features are less important we use in the small branch of the tree
#Feature importance value:
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(rf.feature_importances_) #used to perform feature selection, for unimportant features they will be closed to 0
#return:
array([0. ,0. ,0. ,0. ,0. ,0. ,0. ,0.04,0. ,0.01,0.01, 0. ,0. ,0. ,0. ,0.01,0.01,0. ,0. ,0. ,0. ,0.05, ...
0. ,0.14,0. ,0. ,0. ,0.06,0. ,0. ,0. ,0. ,0. , 0. ,0.07,0. ,0. ,0.01,0. ])
print(sum(rf.feature_importances_))
#return: 1.0

#feature importance as a feature selector:
mask = rf.feature_importances_ > 0.1
print(mask)
#return:array([False, False, ..., True, False])
X_reduced = X.loc[:, mask]
print(X_reduced.columns)
#return:Index(['chestheight', 'neckcircumference', 'neckcircumferencebase', 'shouldercircumference'], dtype='object')

#RFE with RF:
from sklearn.feature_selection import RFE
rfe = RFE(estimator=RandomForestClassifier(),
          n_features_to_select=6, verbose=1)
rfe.fit(X_train,y_train)
#return:
Fitting estimator with 94 features.
Fitting estimator with 93 features
...
Fitting estimator with 8 features.
Fitting estimator with 7 features.

print(accuracy_score(y_test, rfe.predict(X_test)))
#return:0.99

#To speed up the process, we use the "step" parameter
from sklearn.feature_selection import RFE
rfe = RFE(estimator=RandomForestClassifier(),
          n_features_to_select=6, step=10, verbose=1) #step=10:on each iteration the 10 least important features are droped
rfe.fit(X_train,y_train)
#return:
Fitting estimator with 94 features.
Fitting estimator with 84 features.
...
Fitting estimator with 24 features.
Fitting estimator with 14 features.
print(X.columns[rfe.support_]) #print the remaining column name

###Regularized Linear Regession###
##Creating own dataset:with features:y=20+5x1+2x2+0x3+error
##linear regression in Python##
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
# Actual coefficients = [5 2 0]
print(lr.coef_)
[ 4.95  1.83 -0.05] #the third feature has no effect whatsoever
# Actual intercept = 20
print(lr.intercept_)
#return:19.8

#check the accuracy of the model predictions are calculated the R-squared value
# Calculates R-squared,the predictive capacity
print(lr.score(X_test, y_test)) #how much of variance in the target feature our model can predict
#return:0.976

#The model will find the optimal intercept and coefficients by minimizing a loss functions--Mean Squared Error (make the model accurate)
#But if so accurate, it will be overfitting problem.-->Solution:Regularization, try to keep model simple. That is, MSE+alpha*(|beta1|+|beta2|+|beta3|)
#when alpha is too low, leading to overfitting problem. While, if it is too high, it will become too simple and too inaccurate.
#LASSO model: for least absolute shrinkage
from sklearn.linear_model import Lasso
la = Lasso()
la.fit(X_train, y_train)
# Actual coefficients = [5 2 0]
print(la.coef_)
[4.07 0.59 0.  ] #reduce our third feature to 0 to ignore it, also reduce the other coefficients-->lower R^2
print(la.score(X_test, y_test))
0.861
# Create a list that has True values when coefficients equal 0
zero_coef = la.coef_ == 0
# Calculate how many features have a zero coefficient
n_ignored = sum(zero_coef)
print(f"The model has ignored {n_ignored} out of {len(la.coef_)} features.")

#to avoid reducing the R^2:add a value to alpha
from sklearn.linear_model import Lasso
la = Lasso(alpha=0.05)
la.fit(X_train, y_train)
# Actual coefficients = [5 2 0]
print(la.coef_)
[4.07 0.59 0.  ] 
print(la.score(X_test, y_test))
0.974 #the accuracy improved than before

###Combining Feature Selectors###
#we set alpha parameter to find the balance between removing as much features as possible and model accuarcy.
#LassoCV() find the best alpha regressor from sklearn:
linear_model import LassoCV
lcv = LassoCV()
lcv.fit(X_train, y_train)
print(lcv.alpha_)
#return:0.09
mask = lcv.coef_ != 0 #in order to remove the features which equal to 0
print(mask)
#return:[True True False]
reduced_X = X.loc[:, mask]
#Random forest is the combination of decision trees. Weak predictors can combine to form a strong one.We can use this combination idea as follows:

#Feature Selection with lassoCV:
from sklearn.linear_model import LassoCV
lcv = LassoCV()
lcv.fit(X_train, y_train)
lcv.score(X_test, y_test)
print(f'The model explains {r_squared:.1%} of the test set variance')
#return:The model explains 99% of the test set variance.
lcv_mask = lcv.coef_ != 0
sum(lcv_mask)
print(f'{sum(lcv_mask)} features out of {len(lcv_mask)} selected')
#return: 66 features selected.
#Feature selection with random forest:
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
rfe_rf = RFE(estimator=RandomForestRegressor(),
             n_features_to_select=66, step=5, verbose=1)
rfe_rf.fit(X_train, y_train)
rf_mask = rfe_rf.support_
#Feature Selection with gradient boosting:
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingRegressor
rfe_gb = RFE(estimator=GradientBoostingRegressor(),
             n_features_to_select=66, step=5, verbose=1)
rfe_gb.fit(X_train, y_train)
gb_mask = rfe_gb.support_

#combine the feature selectors:
import numpy as np
votes = np.sum([lcv_mask, rf_mask, gb_mask], axis=0)
print(votes)
#return:array([3, 2, 2, ..., 3, 0, 1]). Used to reduce dimensionality and see how a simple linear regressor performs on the reduced dataset.
mask = votes >= 2
reduced_X = X.loc[:, mask]

