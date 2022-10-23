#Environnement

import pandas as pd
import pickle
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from bs4 import BeautifulSoup
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

#Annexes

stop_w = list(set(stopwords.words('english')))
tags_liste = pd.read_csv('tags_liste.csv')
model = pickle.load(open('modele_ovr_tfidf.sav', 'rb'))
model_vect = pickle.load(open('modele_vect_tfidf.sav', 'rb'))
lemmatizer = WordNetLemmatizer()
vocab = pd.read_csv('vocab.csv').values.squeeze().tolist()	
labels = pd.read_csv('labels.csv').values.squeeze().tolist()	

#----------------------------------------------------------------------------------------------------------------------#

#Nettoyage du document

def clean_fct(doc) :									
	doc = BeautifulSoup(doc,features="html.parser").get_text()				
	doc = doc.lower()														
	doc = doc.split()
	print(doc,1)										
	
	doc = [w for w in doc if w not in tags_liste]			
	doc_tags = [w for w in doc if w in tags_liste]			
	doc = ' '.join(doc)										
	print(doc,2)
	
	doc = doc.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
	doc = ''.join([i for i in doc if not i.isdigit()])
	print(doc,3)
	
	doc = word_tokenize(doc)							
	doc = [w for w in doc if w not in stop_w]			
	doc = [w for w in doc if len(w) > 2]				
	print(doc,4)							
    
	doc_lem = []											
	for w in doc :
		doc_lem.append(lemmatizer.lemmatize(w))
	print(doc_lem,5)
	
	doc = [w for w in doc_lem if w in vocab]
	print(doc,6)				
										
	doc = [*doc, *doc_tags]
	print(doc,7)
									
	return ' '.join(doc)

#----------------------------------------------------------------------------------------------------------------------#
	
#Extraction des features par TfIdf
 
def tfidf_fct(doc) :   									
	X_tfidf = model_vect.transform([doc])
	print(X_tfidf,8)
	return X_tfidf
	
#Application du modele regression logistique ovr

def model_fct(X_tfidf) :
	prediction = model.predict(X_tfidf)
	print(prediction,9)
	return prediction

#Identification des tags

def labels_fct(prediction, labels):
    tags_pred = []
    for i, is_label in enumerate(prediction[0]):
        if is_label == 0:
            pass
        else :
            tags_pred.append(labels[i])
    print(tags_pred,10)
    return tags_pred
    
#----------------------------------------------------------------------------------------------------------------------------------# 
  
def predict_fct(doc):
    corpus = clean_fct(doc)
    X_tfidf = tfidf_fct(corpus)
    prediction = model_fct(X_tfidf)
    predicted_tags = labels_fct(prediction, labels)
    return predicted_tags  
  
  

  
  
  
  
  
    
    
	
