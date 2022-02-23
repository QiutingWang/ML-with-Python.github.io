###The Bias VS Variance--Tradeoff###

##Generalization Error##
#find a model f^ achieve a low predictive error on unseen datasets~f---->The Goal:find a model Complexity which minimize GE!
#Two Difficulties of predicting f:
#--Overfitting:f^(x) fits the training set noise(flexibility过强)
#--Underfitting:f^(x) is not flexible enough to approximate f (几乎不能反映data的趋势，flexilibity比较弱) the train set area is roughly equal to the test set error
#Defination of GE:how much do you generlize your unseen data.
#Formular:GE=bias^2+variance+irreducible_error
            #Bias(Accurate):how much average f^!=f; high bias model leads to underfitting
            #Variance(Precise):how much f^ is inconsistant over different trainging sets; high variance model leads to overfitting
            #Irreducible_error is default as a constant
#The Complexity of models-->the flexibility of f^; Increasing maximum tree depth/Minimizing samples per leaf increases the complexity of a decision tree
                        #-->more flexibility,less bias,more variance. As consequence, it is Bias-Variance tradeoff

##Diagnose Bias and Variance Problems##
#Estimate GE is difficult:f is unknown,only use one dataset,noise is unpredictable
#The corresponding solution:1.split train and test data 2.fit f^ use training data 3.evaluate the error of f^ on unseen test set 4.generalize error f^ ~ test set error of f^用testset的error来approximate整个的error
#Evaluate f^ final performance and error,we use cross vaildation(CV) method:K-fold or Hold-out. CV Error=mean of k obtained errors

#Diagnose Variance Problem:if f^ suffer from high variance-->CV Error>training set error-->Overfitting problem-->Solution:complexity↓,more samples gathered
#Diagnose Bias Problem:if f^ suffer from high bias--> CV Error of f^ ~ training set error of f^>>desired error(baselineMSE)-->Underfitting problem-->Solution:complexity↑,gather more features

#K-Fold CV in sklearn using Auto Dataset:
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score #to compute CV
#Set seed for reproducibility
SEED = 123
#Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                                    random_state=SEED)
#Instantiate decision tree regressor and assign it to 'dt'
dt = DecisionTreeRegressor(max_depth=4,
                           min_samples_leaf=0.14,
                           random_state=SEED)
#Evaluate the list of MSE ontained by 10-fold CV
#Set n_jobs to -1 in order to exploit all CPU cores in computation
MSE_CV = - cross_val_score(dt, X_train, y_train, cv= 10,
                           scoring='neg_mean_squared_error',
                           n_jobs = -1)   #cross_val_score does not allow to compute the MSE directly,so we need add a negative sign
#Fit 'dt' to the training set
dt.fit(X_train, y_train)
#Predict the labels of training set
y_predict_train = dt.predict(X_train)
#Predict the labels of test set
y_predict_test = dt.predict(X_test)
#CV MSE
print('CV MSE: {:.2f}'.format(MSE_CV.mean()))
CV MSE: 20.51
#Training set MSE
print('Train MSE: {:.2f}'.format(MSE(y_train, y_predict_train)))
Train MSE: 15.30
#Test set MSE
print('Test MSE: {:.2f}'.format(MSE(y_test, y_predict_test)))
Test MSE: 20.92
#baseline_RMSE serves as the baseline RMSE above which a model is considered to be underfitting and below which the model is considered 'good enough'.

##Ensemble Learning##集成学习
#Limitations of CARTs:orthogonal boundaries,sensitive to small variance in the training set,high variance and overfitting problems---->solution:ensemble learning
#Ensumble Learning-Definition:train different models on the same dataset,then do the predictions for each model,aggregate predictions of all models(Meta-Model),make a final prediction for more robust and less prone to errors.Some of errors will offset with each other
#EG:Ensemble learning in Voting Classifer--binary classification task
# Import functions to compute accuracy and split data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Import models, including VotingClassifier meta-model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import VotingClassifier
# Set seed for reproducibility
SEED = 1
# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size= 0.3,
                                                    random_state= SEED)
# Instantiate individual classifiers
lr = LogisticRegression(random_state=SEED)
knn = KNN()
dt = DecisionTreeClassifier(random_state=SEED)
# Define a list called classifier that contains the tuples (classifier_name, classifier)
classifiers = [('Logistic Regression', lr),
               ('K Nearest Neighbours', knn),
               ('Classification Tree', dt)]
# Iterate over the defined list of tuples containing the classifiers write a for loop:
for clf_name, clf in classifiers:
    #fit clf to the training set
    clf.fit(X_train, y_train)
    # Predict the labels of the test set
    y_pred = clf.predict(X_test)
    # Evaluate the accuracy of clf on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy_score(y_test, y_pred)))
# Instantiate a VotingClassifier 'vc'
vc = VotingClassifier(estimators=classifiers)
# Fit 'vc' to the traing set and predict test set labels
vc.fit(X_train, y_train)
y_pred = vc.predict(X_test)
# Evaluate the test-set accuracy of 'vc'
print('Voting Classifier: {.3f}'.format(accuracy_score(y_test, y_pred))) #the final result,the meta accuracy is higher than any of individual sample's
