import pandas as pd
import numpy as np
import re
import seaborn as sns; sns.set()
import math
import nltk
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
from collections import defaultdict

df = pd.read_csv('data.csv')
print(df)


df['sentiment'].replace({'positive':1, 'negative':0}, inplace=True)
df.head()
print(df)


# functions to remove noise
# remove html tags
def clean_html(text):
 clean = re.compile('<.*?>')
 return re.sub(clean, '', text)
# remove brackets
def remove_brackets(text):
 return re.sub('\[[^]]*\]', '', text)
# lower the cases
def lower_cases(text):
 return text.lower()
# remove special characters
def remove_char(text):
 pattern = r'[^a-zA-z0–9\s]'
 text = re.sub(pattern, '', text)
 return text
# remove noise(combine above functions)
def remove_noise(text):
 text = clean_html(text)
 text = remove_brackets(text)
 text = lower_cases(text) 
 text = remove_char(text) 
 return text
# call the function on predictors
df['review']=df['review'].apply(remove_noise)

print (df)

# importing from nlptoolkit library

from nltk.corpus import stopwords
# creating list of english stopwords
stopword_list = stopwords.words('english')
# removing the stopwords from review
def remove_stopwords(text):
    # list to add filtered words from review
    filtered_text = []
        # verify & append words from the text to filtered_text list
    for word in text.split():
        if word not in stopword_list:
            filtered_text.append(word)
    # add content from filtered_text list to new variable
    clean_review = filtered_text[:]
    # emptying the filtered_text list for new review
    filtered_text.clear()
    return clean_review
df['review']=df['review'].apply(remove_stopwords)


def join_back(text):
    return ' '.join(text)
df['review'] = df['review'].apply(join_back)

print (df)



reviews = df['review'].values
labels = df['sentiment'].values

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=8)
# vectorizing words and storing in variable X(predictor)
X = cv.fit_transform(reviews).toarray()
vocab = cv.get_feature_names_out()

print(vocab)
print (X)
print(labels)

Z_word_counts ={}
word_counts = {}
for l in range(2):
    n_label_items={}
    word_counts[l] = defaultdict(lambda: 0) 
    Z_word_counts[l] = defaultdict(lambda: 0) 
    
for l in range (2):
    n_label_items[l] = X[np.where(labels == l)]
    


for i in range(X.shape[0]):
    y = labels[i]  
    for j in range(len(vocab)):
        if(X[i][j]==1):
            word_counts[y][vocab[j]] += 1
        else:
            Z_word_counts[y][vocab[j]] += 1
 
print (word_counts)
print(Z_word_counts)
   
for i in range(2): 
    for j in range(len(vocab)):
        print("Conditional Propability of(" , vocab[j], "=1| Label=",i,") =", ((word_counts[i][vocab[j]] + 1/2) / (len(n_label_items[i]) + 1)) )
        print("Conditional Propability of(" , vocab[j], "=0| Label=",i,") =", ((Z_word_counts[i][vocab[j]] + 1/2) / (len(n_label_items[i]) + 1)) )
# print(word_counts)
# print(y)


def group_by_label(x, y, labels):
    data = {}
    for l in y:
        data[l] = x[np.where(y == l)]
    return data  

def fit(x, y, labels):
    grouped_data = group_by_label(x, y, labels)
    n_label_items = {}
    log_label_priors = {}  
    for l, data in grouped_data.items():
        n_label_items[l] = len(data)
        log_label_priors[l]= (n_label_items[l]) / len(x)
   
    print(n_label_items, log_label_priors)   
    return n_label_items, log_label_priors

def predict(n_label_items, vocab, word_counts, log_label_priors, labels, x):
    result = []
    e_terms = np.zeros(2, dtype=np.float32) 
    count=1
    for text in x:
        label_scores = {l: log_label_priors[l] for l in labels}
        words = set(w_tokenizer.tokenize(text))
        for word in words:
            if word not in vocab: continue
            for l in labels:
                log_w_given_l= ((word_counts[l][word] + 1/2) / (n_label_items[l] + 1))
                label_scores[l] *= log_w_given_l
                e_terms[l] =  label_scores[l]
        evidence = np.sum(e_terms)
        probs = np.zeros(2, dtype=np.float32)
        for k in range(2):
            probs[k] = e_terms[k] / evidence
              
        print("probabilities of test sentence number ", count)
        print(probs)
        result.append(np.argmax(e_terms))
        count+=1       
    return result

test=['good food but expensive' ,'terrible food but good atmosphere']    
labels =[0,1]    
n_label_items, log_label_priors=fit(reviews,df['sentiment'].values,labels)
pred = predict(n_label_items, vocab, word_counts, log_label_priors, labels, test)
print('Predicted Probabilities on test set: ', pred)

