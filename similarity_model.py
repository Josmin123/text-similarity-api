import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

df=pd.read_csv('DataNeuron_Text_Similarity.csv')
print(f"Loaded {len(df)} text pair.")


model = SentenceTransformer('all-MiniLM-L6-v2')

# Batch encode all texts
text1_list = df['text1'].tolist()
text2_list = df['text2'].tolist()

# Get embeddings in one call each
embeddings1 = model.encode(text1_list, batch_size=64, show_progress_bar=True)
embeddings2 = model.encode(text2_list, batch_size=64, show_progress_bar=True)

similarities = [cosine_similarity([emb1], [emb2])[0][0] for emb1, emb2 in zip(embeddings1, embeddings2)]
df['similarity_score'] = similarities


# Show top 5 most similar
print("\nTop 5 most similar pairs:")
print(df.sort_values(by='similarity_score', ascending=False)[['text1', 'text2', 'similarity_score']].head())

# Show 5 least similar
print("\nTop 5 least similar pairs:")
print(df.sort_values(by='similarity_score', ascending=True)[['text1', 'text2', 'similarity_score']].head())
