###ProcessingData###

#using dummy variables to deal with categorical features:0-->not in this categories;1-->in this categories
#create dummy variables use python:
scikit-learn:OneHotEncoder()
pandas:get_dummies()
#encoding dummy variables
import pandas as pd
df = pd.read_csv('auto.csv') #import the data from csv file
df_origin = pd.get_dummies(df)
print(df_origin.head())
#linear regression with dummy variables:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.3, random_state=42)
ridge = Ridge(alpha=0.5, normalize=True).fit(X_train,y_train) #fit the data
ridge.score(X_test, y_test) #compute the R^2

##Handling the missing data##
#drop any missing data#
df = df.dropna()
#imputing missing data#--make the educating guess about the missing value
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)  #axis=0-->impute along columns;axis-->1 mean rows
imp.fit(X)
X = imp.transform(X) #transform our data,using transform method.The imputers are known as transformers

#imputing within a pipeline#
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
logreg = LogisticRegression() #set the regressor 
steps = [('imputation', imp),('logistic_regression', logreg)]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)

pipeline.fit(X_train, y_train) #fit the pipeline to the train test
y_pred = pipeline.predict(X_test)
pipeline.score(X_test, y_test)

##Centering(normalizing) and Scaling##--we want the features in a similar scale
#Normalize: 1.(xi-x_bar)/variance~N(0,1)
#           2. (xi-minimum(xi))/range
#Scale features:
from sklearn.preprocessing import scale
X_scaled = scale(X)
#check the mean,std and scaled mean,scaled std
np.mean(X), np.std(X)
np.mean(X_scaled), np.std(X_scaled)

#Scaling in a pipeline#
from sklearn.preprocessing import StandardScaler
steps = [('scaler', StandardScaler()),('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=21)
knn_scaled = pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy_score(y_test, y_pred)
#without scaling
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)
knn_unscaled.score(X_test, y_test)

#CV and Scaling in a pipline
steps = [('scaler', StandardScaler()),(('knn', KNeighborsClassifier())] 
pipeline = Pipeline(steps)
parameters = {knn__n_neighbors: np.arange(1, 50)} #specify the parameter space by creating the dictionary. Pipeline step name followed by a double underscore,followed by the hyperparameter name
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=21) #split the data into cross-validation and hold-out set
cv = GridSearchCV(pipeline, param_grid=parameters) 
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
#print the best parameter chosen by our gridsearchCV
print(cv.best_params_)
#along with accuracy
print(cv.score(X_test, y_test))
#classifation report
print(classification_report(y_test, y_pred))
  

