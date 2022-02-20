####Non-negative matrix factorization(NMF)####--dimension reduction techique,but easiler to understand than PCA,and all sample features should>=0

#decomposing samples as sums of their parts(images or document)
#using sklearnNMF: fit/transform,n_components=specified number,work with np arrays or csr_matrix

#For example:TF-IDF method:
from sklearn.decomposition import NMF
model = NMF(n_components=2)
model.fit(samples)
nmf_features = model.transform(samples)
#NMF component:dimension of component=demension of sample and >=0
print(model.components_) #show the result
#NMF features:>=0,using to reconstruct the samples
print(nmf_features)
#--->sample reconstrusction:multiply components by feature values, then add up--this is matrix factorization
print(samples[i,:])
print(nmf_features[i,:])

##NMF learns interpretable parts## but PCA doesn't learn parts.
#For documents,NMF components are topics,features combine and reconstruct topics into documents;
#For images,components are parts of images,represent patterns that frequency occur in the image;
#Grayscale image:measures pixel像素 brightness,⊂[0,1].0 is totally black,1 is totally white.---->convert into flat arrays（row:image,column:pixel)
#visualize samples:
print(sample)
bitmap = sample.reshape((2, 3)) #specify the dimension of original image as a tuple
print(bitmap)
#display the corresponding image
from matplotlib import pyplot as plt
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.show()

##Building recommender system using NMF##
from sklearn.decomposition import NMF
nmf = NMF(n_components=6)
nmf_features = nmf.fit_transform(articles) #given by the column and new array
#how to compare articals using their nmf features? Sometimes meaningless chatter reducing the frequency of topic words: Strong VS Weak tone;But all versions lie on the same line through origin
#then we use angle to compare these lines--'cosine similarity'-->high values, more similar⊂[0,1]
#compute cosine similarity
from sklearn.preprocessing import normalize
norm_features = normalize(nmf_features)
# if has index 23
current_article = norm_features[23,:] #select the row of the corresponding to current article
similarities = norm_features.dot(current_article) #pass it to the dot method of the array of all normalized features
print(similarities) #get the cosine similarity

#label the similarities with article title using DataFrame---normalize and dot()
import pandas as pd
norm_features = normalize(nmf_features)
df = pd.DataFrame(norm_features, index=titles) #rows is normalize features
current_article = df.loc['Dog bites man']
similarities = df.dot(current_article) #calculate the cosine similarity using dot method
print(similarities.nlargest()) #finding the aritical with highest similarity
