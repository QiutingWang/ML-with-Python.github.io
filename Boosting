###Boosting###

#In boosting, ensemble method combining several weak learners to form a strong learner.It trains an ensemble of predictors sequentially.
#Weak learner:model doing slightly better than random guessing.For example,decision stump in CART whose max_depth=1.
#Most popular boosting method:Adaboost(Adaptive boosting),Gradient Boosting...

##Adaboost##

#Definiton of adaboost: (X,y)-->training-->predictor1-->predict-->𝛼1=𝜂*𝛼1-->(W(1),X,y)-->training-->predictor2-->predict-->𝛼2=𝜂*𝛼2-->(W(2),X,y)-->...
#each predictor pays more attention to the instances wrongly predicted by its predecessor,achieving by changing the weights of training samples
#each predictor is assigned by a coefficient 𝛼 that weights its contribution in the ensemble's final prediction,depends on the predictor's training error
#Adaboost training:if there exists n predictors in total.Predictor1 is trained on the inital dataset(X,y), and the training error for predictor1 is determined
#-->the training error can be used to determine 𝛼1 which is predictor1's coefficient in the prediction duration
#-->𝛼1 is used to determine the weights W(2) of the training instances for predictor2-->repeated sequentially,until N predictors forming
#in training is 'Learning Rate' 0<𝜂<=1 (eta)-->used to shrink the 𝛼 of a trained predictor.EG: 𝛼1=𝜂*𝛼1
#The trade off between 𝜂 and # of estimators. Because a smaller 𝜂 can be compensated by larger # of estimators,in order to get a certain performance.

#Prediction:the individual predictors need not to be CARTs,but because of high variance,CARTs are usually used.
#Adaboost for Classification: AdaBoostClassifier()
#Adaboost for Regression: AdaBoostRegressor()

# Import models and utility functions
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# Set seed for reproducibility
SEED = 1
# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    stratify=y,
                                                    random_state=SEED)
# Instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=1, random_state=SEED)
# Instantiate an AdaBoost classifier 'adab_clf'
adb_clf = AdaBoostClassifier(base_estimator=dt, n_estimators=100)
# Fit 'adb_clf' to the training set
adb_clf.fit(X_train, y_train)
# Predict the test set probabilities of positive class
y_pred_proba = adb_clf.predict_proba(X_test)[:,1]  #by passing X_test as a parameter and extract these probabilities by slicing all the values in the second column
# Evaluate test-set roc_auc_score
adb_clf_roc_auc_score = roc_auc_score(y_test, y_pred_proba)
# Print adb_clf_roc_auc_score
print('ROC AUC score: {:.2f}'.format(adb_clf_roc_auc_score))
ROC AUC score: 0.99



##Gradient Boosting##(GB)

#Definiton: (X,y)-->training-->tree1-->predict-->r1=y1-y_hat1-->(X,𝜂r1)-->training-->tree2-->predict-->r2=r1-r_hat1-->(X,𝜂r2)-->...-->rn=r(n-1)-r_hat(n-1)
#not tweak the weights of training instances,comparing with Adaboosting.(没有𝛼)
#Instead, each predictor is trained using 'Residual Errors' of predecessor as labels
#CARTs is used as a base learner.
#The predictions y_hat are used to determine the training set residual errors(r).
#Shrinkage:the prediction of each tree in the ensemble is shrinked after *𝜂
#The trade off between 𝜂 and # of estimators.

#Prediction:
#GB in Regression:y_pred=y1+𝜂*r1+𝜂*r2...𝜂*rn.using GradientBoostingRegressor()
#GB in Classification:GradientBoostingClassifier()

# Import models and utility functions
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
# Set seed for reproducibility
SEED = 1
# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                                    random_state=SEED)
# Instantiate a GradientBoostingRegressor 'gbt'
gbt = GradientBoostingRegressor(n_estimators=300, max_depth=1, random_state=SEED)
# Fit 'gbt' to the training set
gbt.fit(X_train, y_train)
# Predict the test set labels
y_pred = gbt.predict(X_test)
# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2)
# Print the test set RMSE
print('Test set RMSE: {:.2f}'.format(rmse_test))


##Stochastic Gradient Boosting##(SGB)

#Definition:all the training instances(X,y)-->(X_sampled,y_sampled)-->training-->Tree1(feature split pointf1,f2,..fn)-->predict-->r1=y1-y_hat1-->(X,𝜂r1)-->...
#each tree is trained on a random subset of rows of the training data.The samples (40%-80% of the training set) are sampled without replacement
#Features are sampled without replacement when choosing split points-->diversity increases-->further variance increases

# Import models and utility functions
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
# Set seed for reproducibility
SEED = 1
# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                                    random_state=SEED)
# Instantiate a stochastic GradientBoostingRegressor 'sgbt'
sgbt = GradientBoostingRegressor(max_depth=1,
                                 subsample=0.8,  #for each tree to sample 80% of the data for training
                                 max_features=0.2, 
                                 n_estimators=300,
                                 random_state=SEED)
# Fit 'sgbt' to the training set
sgbt.fit(X_train, y_train)
# Predict the test set labels
y_pred = sgbt.predict(X_test)
# Evaluate test set RMSE 'rmse_test'
rmse_test = MSE(y_test, y_pred)**(1/2)
# Print 'rmse_test'
print('Test set RMSE: {:.2f}'.format(rmse_test))
Test set RMSE: 3.95
