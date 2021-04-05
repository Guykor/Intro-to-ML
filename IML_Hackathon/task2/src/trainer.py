from dataloader import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import _pickle as cPickle

class Trainer():
    def __init__(self):
        self.train_set = DataLoader("github_data/all_data/").get_dataset()

    def train(self):
        RF_clf = Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 1))),
            ('tfidf', TfidfTransformer(use_idf=False)),
            ('clf', RandomForestClassifier(max_depth=3000))
        ])

        RF_clf.fit(self.train_set.text, self.train_set.project_name)
        return RF_clf




if __name__ == "__main__":
    t = Trainer()
    t.train()