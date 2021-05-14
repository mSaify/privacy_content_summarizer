from model_training.predict import predict
from  data_collection.create_neutral_datasets import get_sentences_for_file
from model_training.embeddings.embeddings_executor import generate_sentence_embeddings
from torch import load
import numpy as np
from sklearn.cluster import KMeans
import torch


class Summary_Extractor():
    sentences = []
    sent_embeddings = []
    model = None
    word_count = 0

    def __init__(self, cnn_model_path=None):
        self.cnn_model_path = cnn_model_path
        self.model = load(self.cnn_model_path)

    def create_sentences_list(self, files_path=None, file_path=None):
        self.sentences=[]
        if files_path:
            for file_path in files_path:
                self.sentences.extend(get_sentences_for_file(file_path))
        elif file_path:
            self.sentences = get_sentences_for_file(file_path)
        else:
            raise Exception("both multiple files_path and single path can't be None")
        self.word_count=0
        for sent in self.sentences:
            self.word_count = self.word_count + len(sent)
        # self.sent_embeddings = [generate_sentence_embeddings(x) for x in self.sentences]

    def __get_scores(self):
        scores = []
        for sent in self.sentences:
            score = predict(sent, model=self.model)
            # print(score[1].item())
            scores.append(score[1].item())

        return scores

    def risk_focused_content_selection(self, budget=5):
        scores = self.__get_scores()
        top_n_indexes = np.argsort(scores)[-budget:]
        top_n_values = [self.sentences[i] for i in top_n_indexes]

        top_n_scores = np.sort(scores)[-budget:]

        print(f"top {budget} sentences score....")
        print(top_n_scores)

        print(f"top {budget} sentences....")
        print(top_n_values)

        res = str.join('.\n', top_n_values)

        # print("===========summary for risk focus content=================")
        # print(res)
        # print("===========summary for risk focus content=================")
        return res

    def _get_top_n_clusters(self, em, clusters=5):

        print(f"clustering top {clusters} risk factor sentences....")
        em = em.permute(1, 2, 0)

        X = torch.cat([x_pool for x_pool in em],
                      dim=-2)

        X = X.permute(1, 0)

        for i in range(1, 10):
            n_kmeans = KMeans(n_clusters=clusters)
            y_kmeans = n_kmeans.fit_predict(X)
            print(y_kmeans)

        return y_kmeans, n_kmeans.labels_

    def coverage_focused_content_selection(self, budget=5, risk_score_threshold=0.7):
        scores = self.__get_scores()

        top_a_id_scores = {}
        result_sent = []

        for idx, score in enumerate(scores):
            if score > risk_score_threshold:
                top_a_id_scores[idx] = score

        top_a_values = [(i, self.sentences[i]) for i, score in top_a_id_scores.items()]

        print(f"values greater than threshold {risk_score_threshold} sentences score....")
        print(top_a_id_scores)


        self.sent_embeddings = [generate_sentence_embeddings(x) for i, x in top_a_values]

        budget = len(top_a_values) if len(top_a_values) < budget else budget
        em = torch.tensor(self.sent_embeddings)
        clustered_values, labels = self._get_top_n_clusters(em, budget)

        clustered_values_dict = {}

        for i, v in enumerate(clustered_values):
            score = top_a_id_scores[top_a_values[i][0]]
            sent = top_a_values[i][1]
            print(f" cluster : {labels[i]}, score : {score} , sent : {sent}")
            if v in clustered_values_dict:
                if clustered_values_dict[v][1] < score:
                    clustered_values_dict[v] = (top_a_values[i][0], score)
            else:
                clustered_values_dict[v] = (top_a_values[i][0], score)

        print(f"-==========================")
        for i, v in clustered_values_dict.items():
            print(f" cluster : {i}, score : {v[1]} , sent : {self.sentences[v[0]]}")
            result_sent.append(self.sentences[v[0]])



        res = '.\n'.join(result_sent)
        # print("===========summary for coverage focus content=================")
        # print(res)
        # print("===========summary for coverage focus content=================")

        return res

