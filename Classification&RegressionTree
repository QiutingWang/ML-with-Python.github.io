##Classification tree in sklearn## if-else question involving one feature and one split-point

#Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
#Import train_test_split
from sklearn.model_selection import train_test_split
#Import accuracy_score
from sklearn.metrics import accuracy_score
#Split the dataset into 80% train, 20% test
X_train, X_test, y_train, y_test= train_test_split(X, y,
                                                   test_size=0.2,
                                                   stratify=y,  #train and test sets to have the same proportion of class labels as unsplit dataset
                                                   random_state=1)
#Instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=1)
#Fit dt to the training set
dt.fit(X_train,y_train)
#Predict the test set labels
y_pred = dt.predict(X_test)
#Evaluate the test-set accuracy
accuracy_score(y_test, y_pred)

##Logistic Regression VS Classification Tree##
#Import LogisticRegression from sklearn.linear_model
from sklearn.linear_model import LogisticRegression
#Instatiate logreg
logreg =LogisticRegression(random_state=1)
#Fit logreg to the training set
logreg.fit(X_train,y_train)
#Define a list called clfs containing the two classifiers logreg and dt
clfs = [logreg, dt]
#Review the decision regions of the two classifiers (you can use plot_labeled_decision_regions() to plot the decision regions of a list containing two trained classifiers.)
plot_labeled_decision_regions(X_test, y_test, clfs) #As result, the liner model's decision boundry is vertical, but CART is not.
##Decision Region:region in feature space where all instances are assigned to one class label;separated by decision boundaries

##Classification Tree Learning##
#Buliding blocks of decision tree:
#Three kinds of nodes(question or prediction):root(0parent,2children),internal node(1parent,2children),leaf(1parent,0children)---->prediction
#Information Gain(IG):aims at maximizing the information gain obtained after each spilt,which in turn minimizes the entropy and best splits the dataset into groups for effective classification
                      #Formular:IG(f,sp)=I(parent)-[(N_left/N)*I_left+(N_right)/N*I_right];f:feature/sp:split-point
                      #Criteria to measure impurity of a node I:gini index,entroy...
                                                              #gini index:income equality (0,1),less gini index is, there exists a more equal income distribution
                      #it calculates the reduction in entropy or surprise from transforming a dataset.used for feature selection
#Classification-Tree Learning:For unconstrained trees
#nodes are grown recursively
#if IG(node)=0,declare the node a leaf
# Instantiate dt_entropy, set 'entropy' as the information criterion
EG:...dt_entropy = DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=1)......


##Decision-Tree for Regression##

#Regression tree in sklearn#
#Import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
#Import train_test_split
from sklearn.model_selection import train_test_split
#Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE
#Split data into 80% train and 20% test
X_train, X_test, y_train, y_test= train_test_split(X, y,test_size=0.2,random_state=3)
#Instantiate a DecisionTreeRegressor 'dt'
dt = DecisionTreeRegressor(max_depth=4,
                           min_samples_leaf=0.1,  #set the stopping condition in which each leaf has to contain at least 10% of the training data
                           random_state=3)
#Fit 'dt' to the training-set
dt.fit(X_train, y_train)
#Predict test-set labels
y_pred =dt.predict(X_test)
#Compute test-set MSE
mse_dt =MSE(y_test, y_pred) #To evaluate the regression tree
#Compute test-set RMSE
rmse_dt =mse_dt**(1/2)
#Print rmse_dt
print(rmse_dt)

#Information criterion for Regression Tree#
I(node)=MSE(node)=(1/N_node) * sum(yi-ypred_node)**2
ypred_node=(1/N_node)sumyi---mean target value
#Prediction:
ypred_leaf=1/n_leaf * sumyi
#the flexibility of regression tree is important to capture the trend of regression:linear regression VS regression tree
# Predict test set labels 
y_pred_lr = lr.predict(X_test)
#Compute mse_lr
mse_lr = MSE(y_test, y_pred_lr)
#Compute rmse_lr
rmse_lr = mse_lr**(1/2)
#Print rmse_lr
print('Linear Regression test set RMSE: {:.2f}'.format(rmse_lr))
#Print rmse_dt
print('Regression Tree test set RMSE: {:.2f}'.format(rmse_dt))
