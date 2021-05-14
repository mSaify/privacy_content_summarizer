import os
from rouge import rouge
import pandas as pd

from model_training.summary_extractor import Summary_Extractor

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

    sg = Summary_Extractor("model_training/cnn_model.pth")
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

    for service, file_paths in services.items():
        print(f"for a service {service}, following files {file_paths}")

        summary_generator(service, file_paths, alpha=0.8, compression_ratio=1 / 64)
