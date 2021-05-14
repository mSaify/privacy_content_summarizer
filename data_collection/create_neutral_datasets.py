from nltk import sent_tokenize, word_tokenize
import re
import string
import os
import pandas as pd

def if_word_less_then(sentence, number=10):
    words = word_tokenize(sentence)
    if len(words) < number:
        return False

    return True


def sentences_list(text):
    if len(text) > 10:
        return sent_tokenize(text)


def prune(sentence):
    if "__We’re sorry, some parts of the " in sentence or \
            "++" in sentence or \
            "**" in sentence or \
            "##" in sentence:
        return None
    else:
        return sentence.translate(string.punctuation)

def get_sentences_for_file(path):

    with open(path, "r+") as fp:
        text = fp.read()
        sentences = sent_tokenize(text)
        sentences_pruned = [prune(s) for s in sentences]
        sentences_pruned = [s for s in sentences_pruned if s]
        sentences_pruned = [s for s in sentences_pruned if if_word_less_then(s)]
        print(sentences_pruned)
        print(len(sentences_pruned))
        return sentences_pruned

if __name__ == "__main__":


    all_folders = os.walk("./datasets/input/", topdown=True, onerror=None, followlinks=False)
    res_list=[]
    #print(all_folders)

    for folder in list(all_folders)[1:]:

        folder_path = folder[0]
        inside_files = folder[2]

        for file in inside_files:

            path = f"{folder_path}/{file}"

            sentences = get_sentences_for_file(path)
            for sentence in sentences:
                res = {}
                res["service"] = str.strip(folder[0].split(" ")[1]).lower()
                res["quoteText"] = sentence
                res["point"] = "neutral"

                res_list.append(res)

    print(res_list)

    df =pd.DataFrame(res_list)

    df.to_csv("../datasets/intermediate_analysis/input_file_neutral.csv")

