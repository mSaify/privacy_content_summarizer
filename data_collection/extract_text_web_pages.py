
import ssl
import html2text
import requests
import pandas as pd
import random

def get_text_from_html(url):
    context = ssl._create_unverified_context()
    html  = requests.get(url)
    text = remove_html_from_text(html.text)
    return text

def remove_html_from_text(text):
    if type(text) == type("str"):
        h=html2text.HTML2Text()
        h.ignore_links=True
        res=h.handle(text)
        return res

    return ""

def get_service_index_file():
    return pd.read_csv('../datasets/index_list.csv')

def get_labelled_dataset():
    services = pd.read_csv('../datasets/labelled_datasets_orig.csv')
    print(services.head())
    return services

def get_only_approved_dataset(file_name=None):
    if file_name is not None:
        df = pd.read_csv(f'../datasets/intermediate_analysis/only_approved/{file_name}.csv')
        return df

    df = pd.read_csv('../datasets/intermediate_analysis/filtered_approved.csv')
    return df

if __name__ == "__main__":
    approved=[]

    for i in range(0, 11000, 1000):
        approved.extend(get_only_approved_dataset(i))

    approved= pd.DataFrame(approved)
    index = get_service_index_file()
    index_list = list(index["approve_idx"])
    #filter_approved = approved.ix[index_list]
    #filter_approved = approved.ix[index_list]
    #filter_approved.to_csv("filtered_approved.csv")
    filter_approved=pd.read_csv("../datasets/intermediate_analysis/filtered_approved.csv")
    res_list=[]
    for idx, item in filter_approved.iterrows():
        res={}
        service_name = item["service"]
        source_url = item["source"]
        source_paths = source_url.split("/")
        if source_paths[-1] is not None and str.strip(source_paths[-1]) != "":
            file_name=str.strip(source_paths[-1].lower())
        elif source_paths[-2] is not None and str.strip(source_paths[-2]) != "":
            file_name=str.strip(source_paths[-2].lower())

        else:
            file_name=random.random()

        res["service_name"]=service_name
        res["file_name"] = file_name
        res["url"] = source_url

        res_list.append(res)

    pd.DataFrame(res_list).to_csv("../datasets/source_info.csv")


