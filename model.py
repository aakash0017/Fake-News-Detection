from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model_name = 'bert-base-nli-mean-tokens'
model = SentenceTransformer(model_name)

Sentences = [
    "Obama addressed the press at white house conference",
    "the press are eager to listen obama's say at the meeting",
    "Obama talked with the press at the presidential white house meet"
]

sentence_embeddings = model.encode(Sentences)
print(cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings[1:]
))