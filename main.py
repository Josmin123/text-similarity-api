from model import load_model
from similarity import compute_similarity

model=load_model()
text1="Tsunami hit the coast of japan"
text2="Earthquake caused massive waves in japan"

score=compute_similarity(text1,text2,model)
print(f"similarity score is :{score}")