import _pickle as cPickle
from  trainer import *
"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2020

Auther(s):

===================================================
"""

class GitHubClassifier:
    def __init__(self):
        t = Trainer()
        self.model = t.train()
        self.labels = {"building_tool": 0, "espnet": 1, "horovod": 2,
                       "jina": 3, "PuddleHub": 4, "PySolFC": 5, "pytorch_geometric": 6}

    def classify(self,X):
        """
        Receives a list of m unclassified pieces of code, and predicts for each
        one the Github project it belongs to.
        :param X: a numpy array of shape (m,) containing the code segments (strings)
        :return: y_hat - a numpy array of shape (m,) where each entry is a number between 0 and 6
        0 - building_tool
        1 - espnet
        2 - horovod
        3 - jina
        4 - PuddleHub
        5 - PySolFC
        6 - pytorch_geometric
        """

        prediction = self.model.predict(X)
        prediction = prediction.map(self.labels)
        return prediction


