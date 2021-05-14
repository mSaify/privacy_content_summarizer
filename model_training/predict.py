
from model_training.embeddings.embeddings_executor import generate_sentence_embeddings
import torch
import torch.nn.functional as F

def predict(text, model, max_len=200):

    input = generate_sentence_embeddings(text)
    input = torch.tensor(input).float()
    input=input.unsqueeze(0)
    logits = model.forward(input)

    probs = F.softmax(logits, dim=1).squeeze(dim=0)

    #print(f"score {probs[1]} * 100:.2f")

    return probs

