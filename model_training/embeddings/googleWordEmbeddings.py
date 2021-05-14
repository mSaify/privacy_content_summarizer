import pickle
from datetime import datetime
from os.path import basename
from struct import unpack
import os


def loadFile(file_path):
    with open(file_path, "rb") as fp:
        data = pickle.load(fp)
    return data


class EmbeddingsDict(dict):

    def __init__(self, file_path, lang, dim):
        self.lang = lang
        self.dim = dim
        data = loadFile(file_path)
        super().__init__(data)

    def __getitem__(self, key):
        return unpack(str(self.dim)+'e', dict.__getitem__(self, key))


class EmbeddingsEnUSDict(EmbeddingsDict):
    print(f'Enter: {basename(__file__)} EmbeddingsEnUSDict')
    _instance = None

    def __new__(cls, file_path, lang, dim):
        if not cls._instance:
            start = datetime.now()
            cls._instance = EmbeddingsDict(file_path, lang, dim)
            print(f"Loading en-US embeddings took {(datetime.now() - start).seconds} secs")
            print(f'Exit: {basename(__file__)} EmbeddingsEnUSDict')
            return cls._instance
        return cls._instance


class EmbeddingsDeDEDict(EmbeddingsDict):
    print(f'Enter: {basename(__file__)} EmbeddingsDeDEDict')
    _instance = None

    def __new__(cls, file_path, lang, dim):
        if not cls._instance:
            start = datetime.now()
            cls._instance = EmbeddingsDict(file_path, lang, dim)
            print(f"Loading de-DE embeddings took {(datetime.now() - start).seconds} secs")
            print(f'Exit: {basename(__file__)} EmbeddingsDeDEDict')
            return cls._instance
        return cls._instance


def embeddings():
    generic_vectors_dim = {}
    generic_vectors = {}
    WORKING_DIRECTORY = os.getcwd()

    generic_vectors_dim["en-US"] = 300

    return EmbeddingsEnUSDict(os.path.join(WORKING_DIRECTORY, 'datasets/embeddings_data/google_news_w2v_byte.pkl'),
                               'en-US',
                               300)


if __name__ == "__main__":


    print(embeddings()["office"])