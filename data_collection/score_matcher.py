import pandas as pd
from difflib import SequenceMatcher
from extract_text_web_pages import remove_html_from_text
import time


def get_labelled_dataset():
    services = pd.read_csv('../datasets/labelled_datasets_orig.csv')
    print(services.head())
    return services


def get_cases_dataset():
    df = pd.read_json('./datasets/cases.json')
    return df


def get_services_dataset():
    df = pd.read_json('./datasets/services.json')
    # print(df.head())

    return df


def get_only_approved_dataset(file_name=None):
    if file_name is not None:
        df = pd.read_csv(f'../datasets/intermediate_analysis/only_approved/{file_name}.csv')
        return df

    df = pd.read_csv('../datasets/intermediate_analysis/only_approved.csv')
    return df


def generate_only_approved():
    cases = get_flat_cases()
    only_approved = cases.loc[cases['status'] == "approved"]
    only_approved.to_csv('./datasets/only_approved.csv', index=False)


def get_flattened_list_of_points(cases):
    res_list = []
    services = get_services_dataset()
    for index, case in cases.iterrows():

        for point in case['points']:
            res = {}
            res['case_id'] = case['id']
            res['service_id'] = point['service_id']
            res['service'] = services.loc[services['id'] == res['service_id']]['name'].to_string(index=False)
            res['doc_id'] = point['document_id']
            res['classification'] = case['classification']
            res['score'] = case['score']
            res['status'] = point['status']
            res['source'] = point['source']
            res['quoteText'] = point['quoteText']

            res_list.append(res)
    return pd.DataFrame(res_list)


def get_flat_cases():
    df = pd.read_csv('./datasets/flat_cases.csv')
    return df


def similar(a, b):
    if type(a) == type("str") and type(b) == type("str"):
        # print(SequenceMatcher(None,a,b).ratio())
        return SequenceMatcher(None, a, b).ratio()
    return 0.0


# method is compare sentences from approved list to the labeled dataset
def match_quote(only_approved, labelled_dataset):
    total = 0
    res_list = []

    only_approved["index_matched"] = False

    only_approved["service"] = only_approved["service"].str.lower().str.strip()
    labelled_dataset["Service"] = labelled_dataset["Service"].str.lower().str.strip()

    for a_idx, approved in only_approved.iterrows():

        res_l = labelled_dataset.loc[labelled_dataset["Service"] == approved['service']]

        if not res_l.empty:
            for i, rc in res_l.iterrows():
                score1 = similar(remove_html_from_text(approved['quoteText']), rc['QouteText'])
                score2 = similar(approved['quoteText'], rc['QouteText'])
                score = max(score1, score2)

                # take records as similar where score matched is > 80
                if score > 0.80:
                    print(score1)
                    res = {}

                    total = total + 1

                    res["approvedText"] = approved['quoteText']
                    res["labeledText"] = rc['QouteText']
                    res["service"] = approved["service"]
                    res['score'] = score
                    res['approve_idx'] = approved['index']
                    res['label_idx'] = i

                    print(f"approve_idx : {res['approve_idx']}")
                    print(f"label_idx : {i}")

                    res_list.append(res)

                approved["index_matched"] = True

    return res_list


# split dataframe so can be processed in chunks
def split_dataframe_and_save(df, n, folder_path):
    total_rec = len(df)
    idx = 0
    while idx < total_rec:
        res = df[idx:idx + n]
        res.to_csv(f"{folder_path}/{idx}.csv")
        idx = idx + n


if __name__ == "__main__":

    print('base data generator')
    pd.options.display.max_seq_items = None
    pd.options.display.max_seq_items = None
    pd.set_option('display.max_columns', None)

    # labeled_dataset = get_labelled_dataset()
    # cases = get_cases_dataset()
    # cases_flat = get_flattened_list_of_points(cases)
    # print(cases_flat)
    # cases_flat.to_csv('./datasets/flat_cases.csv',index=False)
    # generate_only_approved()

    time_start = time.time()
    res_list = []

    # processed chunked approved list and match it to the labeled dataset.
    for i in range(0, 11000, 1000):
        only_approved = get_only_approved_dataset(i)
        labelled_dataset = get_labelled_dataset()
        res = match_quote(only_approved, labelled_dataset)
        res_list.extend(res)

    final = (pd.DataFrame(res_list))
    final.to_csv("../datasets/index_list.csv")

    print(final[["label_idx", "approve_idx", "labeledText", "service"]])
    print(f"time itt took to execute {time.time() - time_start} in secs")
