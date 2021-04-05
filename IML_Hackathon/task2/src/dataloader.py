import os
import pandas as pd
import numpy as np

DF_NAME = "snippets_df.h5"


class DataLoader:
    """This class handles the retrieval of the raw data from the problem from the
    directory of the project.
    the followiong preprocessing is preformed:
    1. parsing data to rows for each github project
    2. basic cleaning of text
    3. random creation of snippets (all data
    """

    def __init__(self, path):

        self.path = path

    def get_dataset(self):
        try:
            data = pd.read_hdf(self.path + DF_NAME)
            return data

        except(FileNotFoundError):
            data = self.load_data(self.path)
            data = self.random_snippet(data)
            data.to_hdf(self.path + DF_NAME, 'df')
            return data

    def clean_set_indicator(self, data):
        for s in ["train", "validation", "test"]:
            data.project_name = data.project_name.str.replace(f"_{s}.txt", "")
        return data

    def load_data(self, directory):
        dfs = []
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                f = open(directory + filename, "r", encoding='utf-8')
                Lines = f.readlines()
                f.close()
                temp = pd.DataFrame()
                temp["text"] = Lines
                temp["project_name"] = filename
                temp = temp.drop(temp[temp["text"] == "\n"].index).reset_index(drop=True)
                dfs.append(temp)

        data = pd.DataFrame(columns=["text", "project_name"])

        for p in dfs:
            data = pd.concat([data, p], axis=0)
        data = data.reset_index(inplace=False)[["text", "project_name"]]
        data['text'] = data[data.text.str.endswith("\n")]['text'].str[:-2]
        data = self.clean_set_indicator(data)
        return data.dropna()

    def random_snippet(self, data):
        lst = []

        np.random.seed(7)

        for proj in set(data['project_name'].values):
            df = data[data['project_name'] == proj][['text']].reset_index(drop=True)

            i = 0
            snippet_size = np.random.randint(1, 6)
            while i + snippet_size < df.shape[0] - 1:
                lst.append(
                    (df[i:i + snippet_size].apply(lambda x: '\n'.join(x)).to_string()[
                     4:], snippet_size, proj))
                i = i + snippet_size
                snippet_size = np.random.randint(1, 6)

            lst.append(
                (
                    df[i:].apply(lambda x: '\n'.join(x)).to_string()[4:], snippet_size,
                    proj))

        snippetized = pd.DataFrame(lst)
        snippetized.columns = ['text', 'num_lines', 'project_name']

        return snippetized
