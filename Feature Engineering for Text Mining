###Introduction to Text Encoding###Text Mining

##Encoding text##
#free text data--unstructured data-->a series of columns of numbers or vectors
print(speech_df.head())
#removing unwanted characters
speech_df['text'] = speech_df['text'].str.replace('[^a-zA-Z]', ' ') #get the regular expressions,[^a-zA-Z] other non letter characters,replace [^a-zA-Z] with a space
#Before: "Fellow-Citizens of the Senate and of the House of Representatives: AMONG the vicissitudes incident tolife no event could have filled me with greater" .
#After:  "Fellow Citizens of the Senate and of the House of Representatives AMONG the vicissitudes incident to life no event could have filled me with greater"

#standardize the case
speech_df['text'] = speech_df['text'].str.lower()
print(speech_df['text'][0])
#return: "fellow citizens of the senate and of the house of representatives among the vicissitudes incident to life no event could have filled me with greater"...

#length of the text
speech_df['char_cnt'] = speech_df['text'].str.len()
print(speech_df['char_cnt'].head())
#return
0    1889
1     806
2    2408
3    1495
4    2465
Name: char_cnt, dtype: int64
 
#words counts
speech_df['word_cnt'] =speech_df['text'].str.split()
speech_df['word_cnt'].head(1)
#return:['fellow', 'citizens', 'of', 'the', 'senate', 'and',...

speech_df['word_counts'] =speech_df['text'].str.split().str.len()
print(speech_df['word_splits'].head())
#return
0    1432
1     135
2    2323
3    1736
4    2169
Name: word_cnt, dtype: int64
#average length of words
speech_df['avg_word_len']=speech_df['char_cnt'] / speech_df['word_cnt']

##Words count representation##
#we create the table of counts frequency that the words occur in the text
#initializing the vector
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
print(cv)
#specifying the vector
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df=0.1, max_df=0.9) #minimum fraction of the words must occur in,This can be used to remove outlier words that will not generalize across texts. maximum fraction of the words can occur in,This is useful to eliminate very common words that occur in every corpus without adding value such as "and" or "the".
cv.fit(speech_df['text_clean']) #fit the vectorizer
cv_transformed = cv.transform(speech_df['text_clean']) #transform your text
print(cv_transformed)
cv_transformed.toarray()
#getting the features
feature_names = cv.get_feature_names()
print(feature_names) #return a list of the feature generlized 
#another way to combine fit and transforming together:fit_transform() method
cv_transformed = cv.fit_transform(speech_df['text_clean'])
print(cv_transformed)
#putting it all together
cv_df = pd.DataFrame(cv_transformed.toarray(),columns=cv.get_feature_names()).add_prefix('Counts_') #'add_prefix' allows you to distinguish the columns in the future 
print(cv_df.head())
#return:
Counts_aback    Counts_abandoned    Counts_a...
0               1                   0        ...
1               0                   0        ...
2               0                   1        ...
3               0                   1        ...
4               0                   0        ...
#combine and general future analytical models using pandas and concat function
speech_df = pd.concat([speech_df, cv_df],axis=1, sort=False)
print(speech_df.shape)


##TF-IDF Representation##term frequency-inverse document frequency
print(speech_df['Counts_the'].head()) #the counts of the word 'the' shown,to limit these common words from overpowering your model some form of normalization can be used
#Formula: TF-IDF=(count of word occurances/total words in documents) / log(# of docs words is in/total number of docs)
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer()
print(tv)
#max features and stopwards
tv = TfidfVectorizer(max_features=100, #maximum # of columns created,in this case we only use top100 most common words
                     stop_words='english') #lists of common words to omit
tv.fit(train_speech_df['text']) #fit model
train_tv_transformed = tv.transform(train_speech_df['text'])
#put it all together into a dataframe
train_tv_df = pd.DataFrame(train_tv_transformed.toarray(),
                           columns=tv.get_feature_names()).add_prefix('TFIDF_')
train_speech_df = pd.concat([train_speech_df, train_tv_df],axis=1, sort=False)
#inspecting your transforms
examine_row = train_tv_df.iloc[0]  #check which word is recieving the highest scores through the process #if the features being generated make sense or not
print(examine_row.sort_values(ascending=False))  #isolated a single row of the transformed dataframe tv_df
#return:
TFIDF_government   0.367430
TFIDF_public.      0.333237
TFIDF_present.     0.315182
TFIDF_duty.        0.238637
TFIDF_citizens.    0.229644
Name: 0, dtype: float64
#applying the vectorizer to new data,fit the vectorizer only on the training data, and apply it to the test data.
test_tv_transformed = tv.transform(test_df['text_clean']) 
test_tv_df = pd.DataFrame(test_tv_transformed.toarray(), #recreate the test dataset by combining TF-IDF values,feature names,and other columns
                          columns=tv.get_feature_names()).add_prefix('TFIDF_')
test_speech_df = pd.concat([test_speech_df, test_tv_df],axis=1, sort=False)


##N-Grams##
#bag of words  #in this case we create a bi-gram packet;;trigrams: Sequences of 3 consecutive words,which are sequence of n words grouped together. 
tv_bi_gram_vec = TfidfVectorizer(ngram_range = (2,2)) #the value assigned to the argument are the minimum and maximum length of n-grams to be included;ngram_range argument as a tuple (n1, n2)
#or if we haven't create the vectior--Instantiate a trigram vectorizer
cv_trigram_vec = CountVectorizer(max_features=100, 
                                 stop_words='english', 
                                 ngram_range=(2,2))

#Fit and apply bi-gram vectorizer
tv_bi_gram = tv_bi_gram_vec.fit_transform(speech_df['text'])
# Print the bigram features
print(tv_bi_gram_vec.get_feature_names())
#return:
[u'american people', u'best ability ',
 u'beloved country', u'best interests' ... ]

#finding the common words
#Create a DataFrame with the Counts features
tv_df = pd.DataFrame(tv_bi_gram.toarray(),
                     columns=tv_bi_gram_vec.get_feature_names()).add_prefix('Counts_')
tv_sums = tv_df.sum() #checking what are the most common values being recorded
print(tv_sums.head())
#return
Counts_administration government    12
Counts_almighty god                 15
Counts_american people              36
Counts_beloved country               8
Counts_best ability                  8
dtype: int64
#sorting data order
print(tv_sums.sort_values(ascending=False)).head()
#return
Counts_united states         152
Counts_fellow citizens        97
Counts_american people        36
Counts_federal government     35
Counts_self government        30
dtype: int64


