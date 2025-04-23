from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import util

def compute_similarity(text1, text2, model):
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)

    score = util.pytorch_cos_sim(embedding1, embedding2)
    return round(score.item(), 2)
