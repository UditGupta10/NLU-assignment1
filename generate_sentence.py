
# coding: utf-8

# In[13]:


from collections import Counter
import numpy as np
import nltk, re, random, copy
from nltk import word_tokenize, bigrams, trigrams
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords, brown, wordnet
from nltk.stem import WordNetLemmatizer


# In[14]:


train_set = []
dev_set = []
test_set =[]
for c in brown.categories():
    temp = brown.fileids(c)
    temp_length = len(temp)
    train_set += temp[: int(np.ceil(0.6*temp_length))]
    dev_set += temp[int(np.ceil(0.6*temp_length)) : int(np.ceil(0.8*temp_length))]
    test_set += temp[int(np.ceil(0.8*temp_length)):]
    #test_set += temp[-1:]


# In[15]:


brown_sent_train = brown.sents(train_set)
brown_words_train = brown.words(train_set)
#brown_words_train=list(filter(lambda a: a not in ("``", "''", "--", ".", ",", "!",";","(",")","?",":"), brown_words_train))
brown_words_train=list(filter(lambda a: a not in ("``","''", ".",",",";","--","?"), brown_words_train))
brown_words_train = [x.lower() for x in brown_words_train]
brown_words_train += ['<s>', '</s>']*len(brown_sent_train)
brown_unigram_dict_train1 = nltk.FreqDist(brown_words_train)
brown_unigram_dict_train = copy.deepcopy(brown_unigram_dict_train1)
c=0
for (k,v) in brown_unigram_dict_train1.items():
    if(v<=2):
        c+=1
        brown_unigram_dict_train['<unk>'] = brown_unigram_dict_train.pop(k)
        brown_unigram_dict_train['<unk>'] = c


# In[16]:



brown_sents_train = []
for sent3 in brown_sent_train:
    #sent3 = list(filter(lambda a: a not in ("``", "''", "--", ".", ",", "!",";","(",")","?",":"), sent3))
    sent3 = list(filter(lambda a: a not in ("``","''",".",",",";","--","?"), sent3))
    sent3 = [x.lower() for x in sent3]
    sent3 = ['<unk>' if x not in brown_unigram_dict_train.keys() else x for x in sent3]
    brown_sents_train.append(sent3)

# list of sentences (as lists)
elist = []
elist_10 = []
for sent in brown_sents_train:
    elist.append(list(bigrams(sent, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')))
    elist_10.append(list(trigrams(sent, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')))

# list of tuples containing bigrams, trigrams
elist_2 = []
elist_12 = []
for l in elist:
    for t in l:
        elist_2.append(t)
for l in elist_10:
    for t in l:
        elist_12.append(t)
elist_2 += [('<s>', '<s>'), ('</s>', '</s>')]*len(brown_sents_train)
brown_bigram_dict_train = nltk.FreqDist(elist_2)
brown_trigram_dict_train = nltk.FreqDist(elist_12)


# # Generate sentence

# In[5]:


discount=0.5
sent_rand = ['<s>', '<s>']
#cnt=0
for i in range(2,12):
    max_prob=0.0
    for num in range(500):
        word_rand = random.sample(list(filter(lambda a: a not in ('<s>','</s>','<unk>'), brown_unigram_dict_train.keys())),1)
        #print(word_rand)
        #sent_rand = [x.lower() for x in sent_rand]
        sent = sent_rand + word_rand

        unique_1_2 = unique_2_3 = unique_bi_1_2 = unique_bi_2_3 = unique_uni_2_3 = 0
        for k in brown_trigram_dict_train.keys():
            if(k[0] == sent[i-2] and k[1] == sent[i-1]):
                unique_1_2+=1
            if(k[1] == sent[i-1] and k[2] == sent[i]):
                unique_2_3+=1
        for ky in brown_bigram_dict_train.keys():
            if(ky[1] == sent[i-1]):
                unique_bi_1_2+=1
            if(ky[0] == sent[i-1]):
                unique_bi_2_3+=1
            if(ky[1] == sent[i]):
                unique_uni_2_3+=1
        
        
        lambda_0 = 0
        if((sent[i-2], sent[i-1], sent[i]) in brown_trigram_dict_train):
            triple = (sent[i-2], sent[i-1], sent[i])
            lambda_2 = discount*unique_1_2/brown_bigram_dict_train[(sent[i-2], sent[i-1])]
            lambda_1 = discount*unique_bi_2_3/unique_bi_1_2
            prob = (max(brown_trigram_dict_train[triple]-discount, 0)/brown_bigram_dict_train[(sent[i-2], sent[i-1])]) + lambda_2*(max(unique_2_3-discount, 0)/unique_bi_1_2 + lambda_1*(max(unique_uni_2_3-discount,0)/len(brown_bigram_dict_train.keys()) + lambda_0/len(brown_unigram_dict_train.keys())))
            #log_prob += np.log2(prob)
        elif((sent[i-1], sent[i]) in brown_bigram_dict_train):
            double = (sent[i-1], sent[i])
            lambda_1 = discount*unique_bi_2_3/brown_unigram_dict_train[sent[i-1]]
            prob = (max(brown_bigram_dict_train[double]-discount, 0)/brown_unigram_dict_train[sent[i-1]]) + lambda_1*(max(unique_uni_2_3-discount,0)/len(brown_bigram_dict_train.keys()) + lambda_0/len(brown_unigram_dict_train.keys()))
            #log_prob += np.log2(prob)
        elif(sent[i] in brown_unigram_dict_train):
            prob = max(unique_uni_2_3-discount,0)/len(brown_bigram_dict_train.keys()) + lambda_0/len(brown_unigram_dict_train.keys())
            #log_prob += np.log2(prob)
        else:
            prob = lambda_0/len(brown_unigram_dict_train.keys())
            #log_prob += np.log2(prob)
        if(prob>max_prob):
            max_prob = prob
            prob_word = word_rand[0]
    sent_rand += [prob_word]


# In[ ]:


sent_rand

