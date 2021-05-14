embedding_dim = 300
from sklearn.model_selection import train_test_split
from model_training.embeddings.googleWordEmbeddings import embeddings
from allennlp.data.tokenizers.token_class import Token
import torch
import numpy as np
from model_training.embeddings.googleWordEmbeddings import embeddings
from allennlp.data.tokenizers.token_class import Token

def elmo_embedding():
    print("x")
    #lm_model_file = "models.tar.gz"
    # lm_embedder = BidirectionalLanguageModelTokenEmbedder(
    #     archive_file=lm_model_file,
    #     bos_eos_tokens=None
    # )
    #
    # indexer = ELMoTokenCharactersIndexer()
    # vocab = lm_embedder._lm.vocab
    # character_indices = indexer.tokens_to_indices(tokens, vocab)["elmo_tokens"]
    #
    # # Batch of size 1
    # indices_tensor = torch.LongTensor([character_indices])
    #
    # # Embed and extract the single element from the batch.
    # embeddings = lm_embedder(indices_tensor)[0]
    # print(embeddings)
    # f_df = pd.DataFrame(df["X"])
    #
    # np.savetxt("./datasets/training_embeddings.csv", np.asarray(f_df), delimiter=",", fmt="%s")
    #
    # dfs = pd.read_csv("./datasets/training_embeddings.csv",header=None,names=['A'])
    # val = dfs['A']
    # val = val.iloc[1]
    # val = np.asarray(val)
    # print(type(val))
    # print(val)
    # print(val.shape)

    # sentence = "elmo loves you"

def google_word_2_vec():
    return embeddings()

def generate_sentence_embeddings(sentence="",max_len=200):
    sen_list = []
    if isinstance(sentence,str):
        tokens = [Token(word)  for word in sentence.split()  if isinstance(word,str)]
        embed = google_word_2_vec()

        for idx,t in enumerate(tokens):
            try:
                if idx>=max_len:
                    break

                sen_list.append(embed[t.text])
            except Exception as e:
                #print(e)
                sen_list.append([0.0]*embedding_dim)

        for x in range(len(tokens),max_len):
            #print(x)
            sen_list.append([0.0] * embedding_dim)

    else:
        print('no vector for  a sentence')

        for x in range(0, max_len):
            # print(x)
            sen_list.append([0.0] * embedding_dim)

    return np.asarray(sen_list,dtype=float)
