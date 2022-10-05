###Feature selection I, feature information###

##The Curse of Dimensionality##
#models will overfit badly on high dimensional data;we detect the low quality features and remove them
#separate the feature we want to predict from the ones to train the model on
y = house_df['City']
X = house_df.drop('City', axis=1)
#form 70%-30% train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#create SVM classifier 
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
#predict
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, svc.predict(X_test)))
#return:0.826 Our model is able to assign 82.6% of unseen houses to the correct city
print(accuracy_score(y_train, svc.predict(X_train)))
#return:0.832. If the accuracy of X_train>>the accuracy of X_test-->the model didn't generalize well,overfitting problems
#the solution: add more features and the numbers of observations

##Feature with missing value or little variance##--individual properties
#creating feature selector
print(ansur_df.shape)
#return:(6068, 94)
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=1)  #set the minimum variance threshold
sel.fit(ansur_df) #fit the model with our dataset
mask = sel.get_support() #T/F whether the variance of features is above the threshold or not
print(mask) #we name the boolean array: mask
#return:array([ True,  True, ..., False,  True])
#reduce the dimensions using the mask
reduced_df = ansur_df.loc[:, mask]
print(reduced_df.shape)
#return:(6068, 93),we reduce one dimension by setting varaiance threshold
#the problem of variance threshold selector method:it is not easy to interpret the features variance values or compare between features

#visualization:
buttock_df.boxplot()
#normalizing variance before the feature selection
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.005)
sel.fit(ansur_df / ansur_df.mean()) #the formula to normalize. After normalization, the variance will be lower-->then we can think that more features with low variance will be deducted
mask = sel.get_support()
reduced_df = ansur_df.loc[:, mask]
print(reduced_df.shape)
#return:(6068, 45)

#missing values:NaN
#counting
pokemon_df.isna().sum() #get the frequency
pokemon_df.isna().sum()/len(pokemon_df) #get the ratio,between 0 and 1
#based on the missing value ratio,we create the mask for features have fewer missing values than a certain threshold
mask = pokemon_df.isna().sum() / len(pokemon_df) < 0.3  #Fewer than 30% missing values = True value
print(mask)
#return:
Name        True
Type 1      True
Type 2     False
Total       True
HP          True
Attack      True
Defense     True
dtype: bool
#apply the mask
reduced_df = pokemon_df.loc[:, mask]
reduced_df.head() #the type2 features is gone now

##Pairwise Correlation##
sns.pairplot(ansur, hue="gender")
#correlation matrix
weights_df.corr()
#visualizing the correlation matrix
cmap = sns.diverging_palette(h_neg=10,
                             h_pos=240,
                             as_cmap=True)
sns.heatmap(weights_df.corr(), center=0,
            cmap=cmap, linewidths=1,
            annot=True, fmt=".2f")
#create a boolean mask
corr = weights_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool)) #ones_like: create a matrix filled with True values with the same dimensions 
                                               #triu:pass to upper triangle,function to non-upper triangle values to false
#return:
array([[ True,  True,  True],
       [False,  True,  True],
       [False, False,  True]])
#after drop duplicates
sns.heatmap(weights_df.corr(), mask=mask,  #the plot will ignore the upper triangle,for the interested part
            center=0, cmap=cmap, linewidths=1,
            annot=True, fmt=".2f")


##Removing highly correlated features##to avoid models overfit on the small,probably meaningless,differences between these values
#Create positive correlation matrix
corr_df = chest_df.corr().abs() #take the absolute value,to filter strong negative correlations
#Create and apply mask
mask = np.triu(np.ones_like(corr_df, dtype=bool))
tri_df = corr_matrix.mask(mask) #replace all positions in the dataframe where the mask has a True value with NA
tri_df #the results will turns to all the upper trangile value-->True-->NaN
#Find columns that meet treshold
to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.95)] #if any feature stronger than 0.95,drop it!
                                                               #use mask to set half of the matrix to NA values is that we want to avoid removing both features when they have a strong correlations
print(to_drop)
['Suprasternale height', 'Cervicale height']
#Drop those columns
reduced_df = chest_df.drop(to_drop, axis=1)
#correlation != causation
sns.scatterplot(x="N firetrucks sent to fire",
                y="N wounded by fire",data=fire_df)
