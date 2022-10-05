####Dimension Reduction####

###Exploring high dimensional data###
##introduction
#when to use dimensionality reduction?We can drop low variance features.(Dimension-->the columns in the dataframe)
pokemon_df.describe() #we use describe method to discover the std,min,max
pokemon_df.describe(exclude='number') #we will get the summary statistics of non-numeric data
#we should get comfortable with the shape of the dataset and the properties of the features before the dimensional reduction


##Feature Selection and Feature Extraction##
#the benefits of reducing dimensionality:1.dataset less complexity 2.require less disk space 3.less computation time 4.lower chance of model overfitting
#feature selection：
insurance_df.drop('favorite color', axis=1) #move the feature.use axis=1 we specify we drop the column but not the row
sns.pairplot(ansur_df, hue="gender", diag_kind='hist') #pairplot() to visual the small or medium dataset,providing one by one comparsion of each numberic feature in the dataset,in the form of scatter plot,diagonally, a view of distribution of each feature.
                                                       #we set the diagonal is histogram to show the distribution;
                                                       #we use gender category feature to color the plot
#we remove the features are irrelevant or hold little unique information for less information loss

#feature extraction:calculating or extracting new features from the original ones;but new creating features are less intuitive to understand than original ones
#use Principle Component Analysis(PCA) techique


##t-SNE Visualization of high-dimensional data##(t-distributed stochastic neighbor embedding)
#t-SNE maximize the distance in two-dimensional space between observations that are most different in high dimensional space
#check the shape
df.shape #return:(1986,99)
non_numeric = ['BMI_class', 'Height_class',
               'Gender',  'Component', 'Branch']
df_numeric = df.drop(non_numeric, axis=1)
df_numeric.shape #return:(1986,94),remove them before t-sne
#fitting t-SNE
from sklearn.manifold import TSNE
m = TSNE(learning_rate=50) #specify alpha,try different configurations and evaluate these with internal cost function;
                           #high learning rates cause the algorithm to be more adventurous in configurations it tries out
                           #learning rate range (10,1000)
tsne_features = m.fit_transform(df_numeric) #fit and transform
tsne_features[1:4,:] 
#return: 2 dimensions 
array([[-37.962185,  15.066088],
       [-21.873512,  26.334448],
       [ 13.97476 ,  22.590828]], dtype=float32)
#assign t-SNE features into our datasets,assign values to x and y
df['x'] = tsne_features[:,0]
df['y'] = tsne_features[:,1]
#plotting t-SNE
import seaborn as sns
sns.scatterplot(x="x", y="y", data=df)
plt.show()
#coloring points
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x="x", y="y", hue='BMI_class', data=df)
plt.show()

