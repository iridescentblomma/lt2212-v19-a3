import os, sys
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
from statistics import mean

# test.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here.

parser = argparse.ArgumentParser(description="Test a maximum entropy model.")
parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int, default=3, help="The length of ngram to be considered (default 3).")
parser.add_argument("datafile", type=str,
                    help="The file name containing the features in the test data.")
parser.add_argument("modelfile", type=str,
                    help="The name of the saved model file.")

args = parser.parse_args()

print("Loading data from file {}.".format(args.datafile))
print("Loading model from file {}.".format(args.modelfile))

print("Testing {}-gram model.".format(args.ngram))


model = load(args.modelfile+".joblib")
data = pd.read_csv(args.datafile, header=None)
classes = data[data.columns[-1]].tolist()
vectors = data.iloc[:, :-1]

accuracy = model.score(vectors, classes)


def calculate_perplexity(model, vectors):
    log = model.predict_log_proba(vectors)
    prob = model.predict_proba(vectors)
    all_entropy = []
    for i in range(len(log)):
        entropy_vect = 0
        for j in range(len(log[i])):
            entropy_vect += (-1*prob[i][j]*log[i][j])
        all_entropy.append(entropy_vect)
    entropy = mean(all_entropy)
    perplexity = 2 ** entropy
    return perplexity


perplexity = calculate_perplexity(model, vectors)


print("Accuracy is: {}".format(accuracy))
print("Perplexity is: {}".format(perplexity))
