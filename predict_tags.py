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
labels = pd.read_csv('labels.csv').values.squeeze().tolist()	

#Nettoyage des données

def clean_fct(doc) :									
	doc = BeautifulSoup(doc).get_text()				#suppression des balises html
	doc = doc.lower()								#suppression des majuscules
	doc = doc.split()										
	
	doc = [w for w in doc if w not in tags_liste]			#creation du document sans mots tags
	doc_tags = [w for w in doc if w in tags_liste]			#extraction des mots tags
	doc = ' '.join(doc)										
	
	doc = doc.translate(str.maketrans(						#suppression ciblée des ponctuations
	string.punctuation, ' '*len(string.punctuation)))
	doc = ''.join([i for i in doc if not i.isdigit()])
	
	doc = word_tokenize(doc)							#tokenisation du document
	doc = [w for w in doc if w not in stop_w]			#suppression des stop words
	doc = [w for w in doc if len(w) > 2]				#suppression des mots courts
    
	doc_lem = []										#lemmatization
	for w in doc :
		doc_lem.append(lemmatizer.lemmatize(w))									
    
	doc = [*doc, *doc_tags]						#réintégration des mots tags
    
	dico_doc = {}							#creation d'un dictionnaire pour le comptage des mots
	for w in doc :
		if w in dico_doc :
			dico_doc[w] += 1
		else :
			dico_doc[w] = 1
    
	min_freq_w = []							#creation de la liste des mots les moins fréquents
	for w in dico_doc :
		if dico_doc[w] < 3 :
			min_freq_w.append(w)
    
	doc = [w for w in doc if w not in min_freq_w]			#suppression des mots à faible occurence
	return ' '.join(doc)
       
#Creation du corpus

def corpus_fct(texte) :							#nettoyage et concaténation de tous les documents présents dans le texte cible
	corpus = []
	for doc in texte :
		corpus.append(clean_fct(doc))
	return corpus

#---------------------------------------------------------------------------------------------------------------------------------------#
	
#Extraction des features par TfIdf
 
def tfidf_fct(corpus) :   						#vectorisation par tfidf
	X_tfidf = model_vect.transform(corpus)			
	return X_tfidf
	
#Application du modele regression logistique ovr

def model_fct(X_tfidf) :
	prediction = model.predict(X_tfidf)
	return prediction

#Identification des tags
def labels(prediction, labels):
    tags_pred = []
    for i, is_label in enumerate(prediction[0]):
        if is_label == 0:
            pass
        else :
            tags_pred.append(labels[i])
    return tags_pred
    
#--------------------------------------------------------------------------------------------------------------------------------------# 
  
def predict_fct(text):
    corpus = corpus_fct(texte) 						#nettoyage et creation du corpus
    prediction = model_fct(X_tfidf)
    predicted_tags = labels(prediction, labels)				#predictions labélisés
    return predicted_tags  
  
  

  
  
  
  
  
    
    
	
