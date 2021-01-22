# Import all libraries which are required.
from nltk.corpus import stopwords
from nltk import download
import os
from time import time
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import numpy as np
import csv # to be removed later 
import gensim.downloader as api

# Download Stop_words from nltk library
def get_stopwords():
    download('stopwords')
    stop_words = stopwords.words('english')
    return stop_words

# Create a method to import pre-downloaded Embedding matrix [GoogleNews-vectors-negative300.bin.gz]
def chk_for_WrdEmbeddings(filepath):
    if not os.path.exists(filepath):
        raise ValueError("Word-Embedding file not found!")

# Create a temp method to process TextSimilarity Dataset
def get_csv_datafile(filepath):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        sentenceList = list(reader)
    return np.asfarray(sentenceList)

# preprocessing documents
def preprocess_doc(doc):
    processed_doc = doc.lower().split()
    return processed_doc

# Remove stopwords from documents
def remove_stopwords(stop_words, doc):
    _doc = [w for w in doc if w not in stop_words]
    return _doc

# Create Class Word_Mover_Distance
class WordS_Mover_Distance:   
    # create a initializer:
    def __init__(self):
        self.stop_words = get_stopwords()
        # if model already present import;
        if os.path.exists('C:/Users/asus/WMD/model.bin'):
            self.load_model()
        # else     
        else:
            # method to create model
            self.createmodel_usingapi()
            #self.create_model('C:/Users/asus/WMD/GoogleNews-vectors-negative300.bin.gz')
            # method to export model
            #self.export_model()

    # Create model without downloading Embedding-matrix
    # using Genim.downloader api ; arguments - name of model/dataset
    def createmodel_usingapi(self):
        print('downloading awa creating model ')
        self.model = api.load('word2vec-google-news-300')
        
        
    # Create model method
    def create_model(self, filepath):
        print('creating model')
        self.model = KeyedVectors.load_word2vec_format(filepath, binary=True)
        
    # Export model method
    def export_model(self):
        model = self.model
        model.save('C:/Users/asus/WMD/model.bin')

    # Load model method
    def load_model(self):
        print("Loading model...")
        self.model = Word2Vec.load("C:/Users/asus/WMD/model.bin")

    # Text Similarity
    def TextSimilarity(self, s1, s2):
        _model = self.model
        s1 = preprocess_doc(s1)
        s2 = preprocess_doc(s2)
        s1 = remove_stopwords(self.stop_words, s1)
        s2 = remove_stopwords(self.stop_words, s2)
        dist = _model.wmdistance(s1, s2)
        return dist


#wmdobj = WordS_Mover_Distance()

#dist = wmdobj.TextSimilarity("On first day as president, Biden to issue 17 executive actions on COVID, climate change, immigration and more", "Biden to sign 17 executive actions reversing Trump policies on climate, Covid, immigration")
#print(dist)
