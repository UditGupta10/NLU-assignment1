
# coding: utf-8

# In[80]:


import numpy as np
import random, copy
from nltk import bigrams, trigrams, FreqDist
from nltk.corpus import brown


# In[81]:


print('\nThe generated sentence is : ')


# In[82]:


brown_sent_train = brown.sents()
brown_words_train = brown.words()
#brown_words_train=list(filter(lambda a: a not in ("``", "''", "--", ".", ",", "!",";","(",")","?",":"), brown_words_train))
brown_words_train=list(filter(lambda a: a not in ("``","''", ".",",",";","--","?"), brown_words_train))
brown_words_train = [x.lower() for x in brown_words_train]
brown_words_train += ['<s>', '</s>']*len(brown_sent_train)
brown_unigram_dict_train = FreqDist(brown_words_train)
# brown_unigram_dict_train = copy.deepcopy(brown_unigram_dict_train1)
# c=0
# for (k,v) in brown_unigram_dict_train1.items():
#     if(v<=2):
#         c+=1
#         brown_unigram_dict_train['<unk>'] = brown_unigram_dict_train.pop(k)
#         brown_unigram_dict_train['<unk>'] = c


# In[83]:


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
brown_bigram_dict_train = FreqDist(elist_2)
brown_trigram_dict_train = FreqDist(elist_12)


# In[84]:


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

bigram_1_2_dict = FreqDist(bigram_1_2)
bigram_2_3_dict = FreqDist(bigram_2_3)
unigram_1_dict = FreqDist(unigram_1)
unigram_2_dict = FreqDist(unigram_2)


# In[85]:


len(brown_unigram_dict_train.keys())


# # Generate sentence

# In[86]:


len(brown_unigram_dict_train.keys())


# In[89]:


discount=0.7
sent_rand = ['<s>', '<s>']
word_sel = list(filter(lambda a: a not in ('<s>','</s>','<unk>'), brown_unigram_dict_train.keys()))
#cnt=0
for i in range(2,12):
    max_prob=0.0
    for num in range(15000):
        word_rand = random.sample(word_sel, 1)
        #print(word_rand)
        #sent_rand = [x.lower() for x in sent_rand]
        sent = sent_rand + word_rand
        unique_1_2 = bigram_1_2_dict[(sent[i-2], sent[i-1])]
        unique_2_3 = bigram_2_3_dict[(sent[i-1], sent[i])]
        unique_bi_1_2 = unigram_2_dict[sent[i-1]]
        unique_bi_2_3 = unigram_1_dict[sent[i-1]]
        unique_uni_2_3 = unigram_2_dict[sent[i]]

#         unique_1_2 = unique_2_3 = unique_bi_1_2 = unique_bi_2_3 = unique_uni_2_3 = 0
#         for k in brown_trigram_dict_train.keys():
#             if(k[0] == sent[i-2] and k[1] == sent[i-1]):
#                 unique_1_2+=1
#             if(k[1] == sent[i-1] and k[2] == sent[i]):
#                 unique_2_3+=1
#         for ky in brown_bigram_dict_train.keys():
#             if(ky[1] == sent[i-1]):
#                 unique_bi_1_2+=1
#             if(ky[0] == sent[i-1]):
#                 unique_bi_2_3+=1
#             if(ky[1] == sent[i]):
#                 unique_uni_2_3+=1
        
        
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


# In[90]:


for w in sent_rand[2:]:
    print(w, end=' ')


# In[9]:


input('\nPress a key to exit')

