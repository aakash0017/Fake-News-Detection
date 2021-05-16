import numpy as np
import yake 

kw_extractor = yake.KeywordExtractor()
text = """Former President Barack Obama to make his first appearance on The Late Late Show with James Corden"""
language = "en"
max_ngram_size = 3
deduplication_threshold = 0.9
numOfKeywords = 20
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
keywords = custom_kw_extractor.extract_keywords(text)
for kw in keywords:
    print(kw)
    
tup = [i[1] for i in keywords]
tup_array = np.asarray(tup)
print(np.argmax(tup_array))
print(keywords[16])
