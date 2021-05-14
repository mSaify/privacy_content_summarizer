from model_training.predict import predict
from  data_collection.create_ground_truth_source_list import get_sentences_for_file
from model_training.embeddings.embeddings_executor import generate_sentence_embeddings
from torch import load
import numpy as np
from sklearn.cluster import KMeans
import torch
import os
from rouge import rouge
import pandas as pd


class Summary_Generator():
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


def get_file_paths(base_path=None):
    all_folders = os.walk(base_path, topdown=True, onerror=None, followlinks=False)

    services = {}

    for folder in list(all_folders):
        print(folder[0])

        inside_files = folder[2]
        service_name = str.strip(folder[0].split("/")[-1])
        passed_file_service_name = str.strip(base_path.split("/")[-1])
        service_path = f"{base_path}/{service_name}"

        if service_name == passed_file_service_name:
            service_path = f"{base_path}"

        for file in inside_files:
            if service_name in services:
                services[service_name].append(f"{service_path}/{file}")
            else:
                services[service_name] = [f"{service_path}/{file}"]

    print(services)
    return services


def summary_generator(service, file_paths=None, alpha=0.8, compression_ratio=1 / 64):

    sg = Summary_Generator("model_training/cnn_model.pth")
    sg.word_count=0

    sg.create_sentences_list(file_paths)

    print(f"word count for document ====> {sg.word_count}")

    print(f"1/64 ratio compression word budget {int(sg.word_count / 64)}")
    print(f"1/64 ratio compression sentence budget {int((sg.word_count / 64) / 70)}")

    budget = int((sg.word_count / 64) / 70)

    sent1 = sg.risk_focused_content_selection(budget)
    sent2 = ""
    try:
        sent2 = sg.coverage_focused_content_selection(budget, alpha)

    except Exception as err:
        print("----- no sentences above threshold ---- ")

    print("===========summary for risk focus content=================")
    print(sent1)
    print("===========summary for risk focus content=================")
    ref = get_original_quote_text(service)
    print(f"======= risk focus  ROUGE metric === ")
    print(f"{metric_scores(sent1, ref)}")

    print("===========summary for coverage focus content=================")
    print(sent2)
    print("===========summary for coverage focus content=================")
    print(f"======= coverage focus  ROUGE metric === ")
    print(f"{metric_scores(sent2, ref)}")



def metric_scores(hypothesis, reference):
    if hypothesis and reference:
        scores = rouge.Rouge().get_scores(hypothesis, reference)
        print(scores)


def get_original_quote_text(service_name):
    orig = pd.read_csv("datasets/labelled_datasets_orig.csv")
    rec = orig.loc[orig["Service"] == service_name.lower()]
    rec = rec.loc[(orig["Point"] == "bad")]
    ref = ""

    for i,r in rec.iterrows():
        ref = ref + r["QouteText"]

    return ref


if __name__ == "__main__":

    services = get_file_paths("datasets/held_out_test_data")

    #services = get_file_paths("datasets/held_out_test_data/Brainly")

    for service, file_paths in services.items():
        print(f"for a service {service}, following files {file_paths}")

        summary_generator(service, file_paths, alpha=0.8, compression_ratio=1 / 64)
