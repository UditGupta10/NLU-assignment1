
# coding: utf-8

# In[1]:


import numpy as np
import nltk, random, copy
from nltk import bigrams, trigrams
from nltk.corpus import gutenberg as gb


# In[2]:


temp = gb.fileids()
random.shuffle(temp)
temp_length = len(temp)
train_set = temp[: int(np.ceil(0.6*temp_length))]
dev_set = temp[int(np.ceil(0.6*temp_length)) : int(np.ceil(0.8*temp_length))]
test_set = temp[int(np.ceil(0.8*temp_length)):]


# In[3]:


gb_sent_train = gb.sents(train_set)
gb_words_train = gb.words(train_set)
#gb_words_train=list(filter(lambda a: a not in ("``", "''", "--", ".", ",", "!",";","(",")","?",":"), gb_words_train))
gb_words_train=list(filter(lambda a: a not in ("``","''", ".","[","]",",",":",";","--"), gb_words_train))
gb_words_train = [x.lower() for x in gb_words_train]
gb_words_train += ['<s>', '</s>']*len(gb_sent_train)
gb_unigram_dict_train1 = nltk.FreqDist(gb_words_train)
gb_unigram_dict_train = copy.deepcopy(gb_unigram_dict_train1)
c=0
for (k,v) in gb_unigram_dict_train1.items():
    if(v<=3):
        c+=1
        gb_unigram_dict_train['<unk>'] = gb_unigram_dict_train.pop(k)
        gb_unigram_dict_train['<unk>'] = c


# In[4]:



gb_sents_train = []
for sent3 in gb_sent_train:
    #sent3 = list(filter(lambda a: a not in ("``", "''", "--", ".", ",", "!",";","(",")","?",":"), sent3))
    sent3 = list(filter(lambda a: a not in ("``","''",".","[","]",",",":",";","--"), sent3))
    sent3 = [x.lower() for x in sent3]
    sent3 = ['<unk>' if x not in gb_unigram_dict_train.keys() else x for x in sent3]
    gb_sents_train.append(sent3)

# list of sentences (as lists)
elist = []
elist_10 = []
for sent in gb_sents_train:
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
elist_2 += [('<s>', '<s>'), ('</s>', '</s>')]*len(gb_sents_train)
gb_bigram_dict_train = nltk.FreqDist(elist_2)
gb_trigram_dict_train = nltk.FreqDist(elist_12)


# In[5]:


unique_trigram = list(set(elist_12))
unique_bigram = list(set(elist_2))

bigram_1_2=[]
bigram_2_3=[]
unigram_1 = []
unigram_2 = []
for t in unique_trigram:
    bigram_1_2.append(t[:2])
    bigram_2_3.append(t[1:])
for t in unique_bigram:
    unigram_1.append(t[0])
    unigram_2.append(t[1])

bigram_1_2_dict = nltk.FreqDist(bigram_1_2)
bigram_2_3_dict = nltk.FreqDist(bigram_2_3)
unigram_1_dict = nltk.FreqDist(unigram_1)
unigram_2_dict = nltk.FreqDist(unigram_2)


# # Test

# In[6]:


gb_sents_test1 = [['<s>', '<s>'] + sent + ['</s>', '</s>'] for sent in gb.sents(test_set)]
# list of sentences as lists
gb_sents_test = []
for sent3 in gb_sents_test1:
    #sent3 = list(filter(lambda a: a not in ("``", "''", "--", ".", ",", "!",";","(",")","?",":"), sent3))
    sent3 = list(filter(lambda a: a not in ("``", "''",".","[","]",",",":",";","--"), sent3))
    sent3 = [x.lower() for x in sent3]
    sent3 = ['<unk>' if x not in gb_unigram_dict_train.keys() else x for x in sent3]
    gb_sents_test.append(sent3)


# In[7]:


discount = 0.7
M = 0
log_prob = 0.0
#gb_sents_test_sample = random.sample(gb_sents_test, 20)
for sent in gb_sents_test:
    sent_len = len(sent)
    M += (sent_len-3)
    for i in range(2, sent_len-1):
        unique_1_2 = bigram_1_2_dict[(sent[i-2], sent[i-1])]
        unique_2_3 = bigram_2_3_dict[(sent[i-1], sent[i])]
        unique_bi_1_2 = unigram_2_dict[sent[i-1]]
        unique_bi_2_3 = unigram_1_dict[sent[i-1]]
        unique_uni_2_3 = unigram_2_dict[sent[i]]         
        lambda_0 = 0
        if((sent[i-2], sent[i-1], sent[i]) in gb_trigram_dict_train):
            triple = (sent[i-2], sent[i-1], sent[i])
            lambda_2 = discount*unique_1_2/gb_bigram_dict_train[(sent[i-2], sent[i-1])]
            lambda_1 = discount*unique_bi_2_3/unique_bi_1_2
            prob = (max(gb_trigram_dict_train[triple]-discount, 0)/gb_bigram_dict_train[(sent[i-2], sent[i-1])]) + lambda_2*(max(unique_2_3-discount, 0)/unique_bi_1_2 + lambda_1*(max(unique_uni_2_3-discount,0)/len(gb_bigram_dict_train.keys()) + lambda_0/len(gb_unigram_dict_train.keys())))
            log_prob += np.log2(prob)
        elif((sent[i-1], sent[i]) in gb_bigram_dict_train):
            double = (sent[i-1], sent[i])
            lambda_1 = discount*unique_bi_2_3/gb_unigram_dict_train[sent[i-1]]
            prob = (max(gb_bigram_dict_train[double]-discount, 0)/gb_unigram_dict_train[sent[i-1]]) + lambda_1*(max(unique_uni_2_3-discount,0)/len(gb_bigram_dict_train.keys()) + lambda_0/len(gb_unigram_dict_train.keys()))
            log_prob += np.log2(prob)
        elif(sent[i] in gb_unigram_dict_train):
            prob = max(unique_uni_2_3-discount,0)/len(gb_bigram_dict_train.keys()) + lambda_0/len(gb_unigram_dict_train.keys())
            log_prob += np.log2(prob)
        else:
            prob = lambda_0/len(gb_unigram_dict_train.keys())
            log_prob += np.log2(prob)


# In[9]:


print('Perplexity for GB train GB test: ' + str(2**(-1*log_prob/M)))

