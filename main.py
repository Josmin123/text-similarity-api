from model import load_model
from similarity import compute_similarity

model=load_model()
text1 = "The weather is pleasant today."
text2 = "Today is holiday."


score=compute_similarity(text1,text2,model)
print(f"similarity score is :{score}")