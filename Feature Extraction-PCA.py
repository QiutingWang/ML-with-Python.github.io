####Feature Extraction####

##Intro to PCA##
sns.scatterplot(data=df, x='handlength', y='footlength')
scaler = StandardScaler() #scaler the feature first, the values are easier to compare
df_std = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
#reference point:the center of point cloud
#Vector1:vectors and direction of the strongest pattern; follow this vectors, there exists a positive relationship between dependent and independent variables;
#Vector2:we set another vector which perpendicular to the first one to account for the rest of the variance
#every point of the dataset could be described by multiplying and then summing these two perpendicular vectors
#we create a new reference system aligned with the variance in the data
#the new coordinates in the new reference system called Principle Components建立一个新的坐标系

###Principle Components Analysis###
##Calculating Principle Components##
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
std_df = scaler.fit_transform(df)
#create a PCA instance 2 features
from sklearn.decomposition import PCA
pca = PCA()
print(pca.fit_transform(std_df)) #PCA remove the correlation and no more duplicate information and that they rank from most to least important 
[[-0.08320426 -0.12242952]
 [ 0.31478004  0.57048158]
 ...
 [-0.5609523   0.13713944]
 [-0.0448304  -0.37898246]]
#principle component explained variance ratio#
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(std_df)
print(pca.explained_variance_ratio_) 
#return:array([0.90, 0.10]), means the first component explains 90% of the variance in the data and second remaining 10%-->the second component can be dropped

#PCA for dimension reduction for high dimension example#
pca = PCA()
pca.fit(ansur_std_df)
print(pca.explained_variance_ratio_)
#return:
array([0.44, 0.18, 0.04, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,0.01,0.01,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. , ...
0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,0. ,0. ,0. ,0. ,0. ])
print(pca.explained_variance_ratio_.cumsum())
#return:
array([0.44, 0.62, 0.66, 0.69, 0.72, 0.74, 0.76, 0.77, 0.79, 0.8 , 0.81,
       0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.87, 0.88, 0.89, 0.89, 0.9 ,
       0.9 , 0.91, 0.92, 0.92, 0.92, 0.93, 0.93, 0.94, 0.94, 0.94, 0.95,
       ...
       0.99,0.99,0.99,0.99,0.99,1. ,1. ,1. ,1. ,1. ,1. , 1. , 1. , 1. ,
        1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,1. ,1. ,1. ,1. ,1. ])

###PCA Application###we need to decide how much of the explained variances are willing to sacrifice
#understand the components#
print(pca.components_) #look at the components attribute
#return:
array([[  0.71, 0.71],   #According to the dataset, PC1=0.71*feature1+0.71*feature2
       [ -0.71, 0.71]])
#PCA in a pipeline#
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline #combine two operations into the pipeline
pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reducer', PCA())])
pc = pipe.fit_transform(ansur_df)
print(pc[:,:2])
#return:
array([[-3.46114925,  1.5785215 ],
       [ 0.90860615,  2.02379935],
       ...,
       [10.7569818 , -1.40222755],
       [ 7.64802025,  1.07406209]])
#checking the effect of categorical features# PCA is not a good measure to deal with categoical data
print(ansur_categories.head())

ansur_categories['PC 1'] = pc[:,0]
ansur_categories['PC 2'] = pc[:,1]
sns.scatterplot(data=ansur_categories,
                x='PC 1', y='PC 2',
                hue='Height_class', alpha=0.4)
sns.scatterplot(data=ansur_categories,
                x='PC 1', y='PC 2',
                hue='Gender', alpha=0.4)
sns.scatterplot(data=ansur_categories,
                x='PC 1', y='PC 2',
                hue='BMI_class', alpha=0.4)
#PCA in a model pipeline
pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reducer', PCA(n_components=3)),
        ('classifier', RandomForestClassifier())])
print(pipe['reducer'])
#return:PCA(n_components=3)
pipe.fit(X_train, y_train)
pipe['reducer'].explained_variance_ratio_
#return:array([0.56, 0.13, 0.05])
pipe['reducer'].explained_variance_ratio_.sum()
0.74
print(pipe.score(X_test, y_test))
0.986
  

###Principle Component Selection###
##set the explained variance threshold##
pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reducer', PCA(n_components=0.9))]) #set the Explained variance ratio=0.9
# Fit the pipe to the data
pipe.fit(poke_df)
print(len(pipe['reducer'].components_))
#return:5

#get the optimal number of components:
pipe.fit(poke_df)
var = pipe['reducer'].explained_variance_ratio_
plt.plot(var)
plt.xlabel('Principal component index')
plt.ylabel('Explained variance ratio')
plt.show()
#we can see the most explained variance ratio consentratein the first few components
#normally, we can use the slope of the line to get the concentration of the ratio

#go back from pc to orignial component array: 
pca.inverse_transform(pc)
#(the path we get pc: original data array->fit->transform->pc)


