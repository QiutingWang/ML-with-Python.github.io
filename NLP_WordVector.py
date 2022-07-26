#introduction and word vectors#

#Computer thinks of meaning# wordnet:a thesaurus containing lists of synonym sets and hypernyms.A great resource of NLP
https://wordnet.princeton.edu/
#synonym sets containing "good" 同义词,words mean the same things
!pip install nltk
from nltk.corpus import wordnet as wn  # wordnet:a thesaurus containing lists of synonym sets and hypernyms 
poses={'n':'noun','v':'verb',"s":"adj(s)","a":"adj","r":"adv"}
for synset in wn.synset("good"):
    print("{}:{}".format(poses[synset.pos])),
          ",".join([l.name() for l in synset.lemmas()])

'''return:
noun: good
noun: good, goodness
noun: good, goodness
noun: commodity, trade_good, good
adj: good
adj (sat): full, good
adj: good
adj (sat): estimable, good, honorable, respectable
adj (sat): beneficial, good
adj (sat): good
adj (sat): good, just, upright
...
adverb: well, good
adverb: thoroughly, soundly, good'''

#hypernyms of “panda”：上位词，外延更广的主题词
from nltk.corpus import wordnet as wn
panda = wn.synset("panda.n.01")
hyper = lambda s: s.hypernyms()
list(panda.closure(hyper))
'''return:
[Synset('procyonid.n.01'),
Synset('carnivore.n.01'),
Synset('placental.n.01'),
Synset('mammal.n.01'),
Synset('vertebrate.n.01'),
Synset('chordate.n.01'),
Synset('animal.n.01'),
Synset('organism.n.01'),
Synset('living_thing.n.01'),
Synset('whole.n.02'),
Synset('object.n.01'),
Synset('physical_entity.n.01'),
Synset('entity.n.01')]'''

#Problems about Wordnet#
#missing new meaning of a word, not keep into update
#require human labor for correction and adaption
#can not compute accurate word similarity-->learn to encode similarity

#Representing words into discrete symbols#
#Vector dimension = number of words in vocabulary
#Can be represented by one-hot vectors

#Distributional semantics:
#A word’s meaning is given by the words that frequently appear close-by
#word vectors/word embeddings/neural word representations:distributed representation
#similar to vectors of words that appear in similar contexts, measuring similarity as the vector dot (scalar) product:cosine similarity
#use word cloud for visualization
import matplotlib.pyplot as plt
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(worldplot)
#we want to get the first 200 words
words = [word for word in list(model.key_to_index.keys())[:200]]
labels = [word for word in words]
X = model[words]

#reduce the dimensionality of word vectors using TSNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
low_dim_embs = tsne.fit_transform(X)
#visualizing
plot_with_labels(low_dim_embs, labels)

#Word2vec: a framework for learning words vectors
#calculating the distribution similarity task of predicting well what words will occur in the context of other words
#Center word:c at Position:t,     Context words:o    
#use the similarity of the word vectors for c and o-->Calculate the Pr(o|c) (or vice versa) 
#keep adjusting the word vectors to maximize the Pr
#Fix size window:Pr(𝑤_t+j | 𝑤_t)
#Objective Function:average negative log likelihood, minimize the loss function
#using (stochastic) gradient descent for optimization

!pip install gensim
# download and unzip GoogleNews word2vec
!wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
!gzip -d GoogleNews-vectors-negative300.bin.gz #unzip the file
# load word2vec model
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
print(model["good"])
'''return:
array([ 0.04052734,  0.0625    , -0.01745605,  0.07861328,  0.03271484,
       -0.01263428,  0.00964355,  0.12353516, -0.02148438,  0.15234375,
       -0.05834961, -0.10644531,  0.02124023,  0.13574219, -0.13183594,
        0.17675781,  0.27148438,  0.13769531, -0.17382812, -0.14160156,
       -0.03076172,  0.19628906, -0.03295898,  0.125     ,  0.25390625,
        0.12695312, -0.15234375,  0.03198242,  0.01135254,...])'''

# print the similarity between "good" and "awesome"
model.similarity("good", "awesome")
# Output: 0.52400756

#Compute the most similar words for a given word:
model.most_similar("good")
''' output:
[('great', 0.7291510105133057),
 ('bad', 0.7190051078796387),
 ('terrific', 0.6889115571975708),
 ('decent', 0.6837348341941833),
 ('nice', 0.6836092472076416),
 ('excellent', 0.644292950630188),
 ('fantastic', 0.6407778263092041),
 ('better', 0.6120728850364685),
 ('solid', 0.5806034803390503),
 ('lousy', 0.576420247554779)]
'''
result = model.most_similar(positive=["woman", "king"], negative=["man"])
print("{}: {:.4f}".format(*result[0]))

'''output
queen: 0.7118
'''
#analogy function an pass different combinations of words:
def analogy(x1, x2, y1):
    result = model.most_similar(positive=[y1, x2], negative=[x1])
    print("{}".format(result[0][0]))

analogy("tall", "tallest", "long")
#output: longest