
from sentence_transformers import util

def compute_similarity(text1, text2, model):
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    raw_score = util.pytorch_cos_sim(embedding1, embedding2).item()
    normalized_score = (raw_score + 1) / 2

    return round(normalized_score, 2)


