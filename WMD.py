# Import all libraries which are required.
from nltk.corpus import stopwords
from nltk import download
import os
from time import time
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import numpy as np
import csv # to be removed later 

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
        if os.path.exists('mymodel'):
            self.load_model()
        # else     
        else:
            # method to create model
            self.create_model('GoogleNews-vectors-negative300.bin.gz')
            # self.export_model()
            # method to export model
            #self.export_model()

    # Create model method
    def create_model(self, filepath):
        print('creating model')
        self.model = KeyedVectors.load_word2vec_format(filepath, binary=True)
        
    # Export model method
    def export_model(self):
        model = self.model
        model.save('mymodel')

    # Load model method
    def load_model(self):
        print("Loading model...")
        self.model = Word2Vec.load("mymodel")

    # Text Similarity
    def TextSimilarity(self, s1, s2):
        _model = self.model
        s1 = preprocess_doc(s1)
        s2 = preprocess_doc(s2)
        s1 = remove_stopwords(self.stop_words, s1)
        s2 = remove_stopwords(self.stop_words, s2)
        dist = _model.wmdistance(s1, s2)
        return dist


wmdobj = WordS_Mover_Distance()

dist = wmdobj.TextSimilarity("SpaceX to set a new record of rapid reuse with latest Starlink launch", "With latest Starlink launch, SpaceX to set record for rapid reuse")
print(dist)
