from model import load_model
from similarity import compute_similarity

model=load_model()
text1 = "Apples are fruits."
text2 = "my name is josmin"


score=compute_similarity(text1,text2,model)
print(f"similarity score is :{score}")