from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# from Keyword_extractor import getKeywords


model_name = 'bert-base-nli-mean-tokens'
model = SentenceTransformer(model_name)

class Sentence_Similarity:
    
    def __init__(self, Context):
        
        # context sentence
        self.Context = Context
        self.Context_Embedding = model.encode(self.Context)
        # max_comparisons
        self.max_comparison = 20
        
    def getRefrences(self, list_references):
        # TODO change list_referneces -> list_keyword and allow web scrapper to return top results.
        self.References = list_references
        self.References_Embeddings = model.encode(list_references)
    
    def check_Similarity(self):
        result = cosine_similarity(
            [self.Context_Embedding],
            self.References_Embeddings
        )
        return result 
    
    def getMostSimilar(self):
        similarity_array = self.check_Similarity()
        most_similar = self.References[np.argmax(similarity_array)]
        return most_similar
        
            
if __name__ == "__main__":
    a = Sentence_Similarity('James corden show eagers the first apperance of Former president Obama')
    ref = [
        "Former President Barack Obama to make his first appearance on The Late Late Show with James Corden",
        "Michelle Obama’s advice for coping with depression: ‘Develop your own tools’ and give yourself a break",
        "Barack and Michelle Obama Pay Tribute to Their Family Dog, Bo, Who Died This Weekend",
        "President Barack Obama To Make Debut Appearance On ‘The Late Late Show With James Corden'"
    ]
    a.getRefrences(ref)
    res = a.getMostSimilar()
    print(res)