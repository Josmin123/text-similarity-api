from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(text1,text2,model):
    emdeddings=model.encode([text1,text2])
    score=cosine_similarity([emdeddings[0],emdeddings[1]])[0][0]
    return round(float(score),2)