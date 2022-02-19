###Hierarchical and t-SNE---For visualiztion###
#t-SNE:creates a 2D map of a datasets

##Hierarchical##
##Hierarchical clustering with SciPy##---merging&dendrogram
#Use the linkage() function to obtain a hierarchical clustering of the grain samples, and use dendrogram()层级树图to visualize the result
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
# Calculate the linkage: mergings
mergings = linkage(samples,method='complete') #using the given samples
#Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=varieties,  #the label name，横坐标显示的内容
           leaf_rotation=90,  #垂直角度
           leaf_font_size=6,
)
plt.show()

##Cluster Label in Hierarchical Clustering##--also used in cross-tabulation交叉列表
#intermediate clustering&height on dendrogram横切，看在当前那个高度上，分支的clustering数量有几个
#height on dendrogram=distance between merging clusters
#the distance between 2 clusters is measured using a 'linkage method';'complete' linkage,specify by method parameter-->the max of the distances between the samples

#Extracting cluster labels#--using fcluster() function, it will return a NumPy array for cluster labels
from scipy.cluster.hierarchy import linkage
mergings = linkage(samples, method='complete')
from scipy.cluster.hierarchy import fcluster
labels = fcluster(mergings, 15, criterion='distance') #specify the maximum height is 15
print(labels)

#sort data by clustering label and print the result;aligning cluster labels with country names
import pandas as pd
pairs=pd.DataFrame({'labels':labels,'countries':country_names})
print(pairs.sort_values('labels')) #the labels start by 1, not 0

#In complete linkage, the distance between clusters is the distance between the furthest points of the clusters. 
#In single linkage, the distance between clusters is the distance between the closest points of the clusters.

###t-SNE and 2-dimensional maps###---t-dist stochastic neighbor embedding; it can represent the distance between samples
##t-SNE in sklearn##--fit_transform() method:fit the model and transform the data//learning rates choose: wrong choice-->points bunch together. ⊂[50,200];And different each time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
model = TSNE(learning_rate=100)
transformed = model.fit_transform(samples) #applied the fit transform into samples
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys, c=species) #covering the plot using species
plt.show()





###Visualizing the PCA transformation###

##Demension Reduction##--remove less-informative noise features;
#PCA:'principle component analysis',fundamental dimension reduction technique;rotate data sample to be aligned with axes,with µ=0;
from sklearn.decomposition import PCA
model = PCA()
model.fit(samples)
transformed = model.transform(samples)
#PCA features are not linearly correlated--decorrelation
#pearson correlation⊂[-1,1]
#Principle componenets:direction of variance,PCA aligns principle components with the axes. components_ attribute of PCA object

##Intrinstic Dimension##number of features needed to approximate the dataset
#PCA identifies intrinsic dimension when samples have any number of features; intrinsic dimension=# of pca features with significant variance>=threshold (ignore the unimportant features)
#PCA rotates and shift the samples align them with coordinate axes
#PCA features are ordered by variance descending

#plotting the variances of PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA() #build the model
pca.fit(samples) #fit the model
features = range(pca.n_components_) #create a range enumerating the PCA features,extract the number of components
#make a bar plot
plt.bar(features, pca.explained_variance_) #plot the variance
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()

##Dimension reduction with PCA##informative and noisy
#specify how many features to keep:PCA(n_components=2), to keep only first two PCA features
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(samples)
transformed = pca.transform(samples) #transform the models as usual
print(transformed.shape) #it will return the # of features==2 what we expected

#Word frequency arrays:rows--documents,columns--words. TF-IDF(word frequency) method:how often the word occurs in each document
#Sparse arrays and csr_matrix:Sparse--most entries=0;csr_matrix=only non-zero entries
#TruncatedSVD & csr_matrix: because scikit-learn PCA doesn't support csr_matrix,then should use TruncatedSVD

from sklearn.decomposition import TruncatedSVD
model = TruncatedSVD(n_components=3)
model.fit(documents)        # documents is csr_matrix
TruncatedSVD(algorithm='randomized', ... )
transformed = model.transform(documents)

#For example:Word frequency arrays (For text mining)
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer()
#Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)
#Print result of toarray() method
print(csr_mat.toarray())
#Get the words,The columns of the array correspond to words
words = tfidf.get_feature_names()
#Print words
print(words)
