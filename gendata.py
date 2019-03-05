import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
import nltk
from nltk import ngrams

# gendata.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here. You may not use the
# scikit-learn OneHotEncoder, or any related automatic one-hot encoders.

parser = argparse.ArgumentParser(description="Convert text to features")
parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int, default=3, help="The length of ngram to be considered (default 3).")
parser.add_argument("-S", "--start", metavar="S", dest="startline", type=int,
                    default=0,
                    help="What line of the input data file to start from. Default is 0, the first line.")
parser.add_argument("-E", "--end", metavar="E", dest="endline",
                    type=int, default=None,
                    help="What line of the input data file to end on. Default is None, whatever the last line is.")
parser.add_argument("inputfile", type=str,
                    help="The file name containing the text data.")
parser.add_argument("outputfile", type=str,
                    help="The name of the output file for the feature table.")

args = parser.parse_args()


def get_tokens(file):
    with open(file, "r", encoding="UTF8") as f:
        text = f.read()
        raw_tokens = text.split(" ")
        tokens_POS = []
        for t in raw_tokens:
            tokens_POS.append((t.split("/")))
    tokens = []
    for t_p in tokens_POS:
        tokens.append(t_p[0])
    return tokens[:10]


def create_ngrams(tokens, n):
    n_grams = ngrams(tokens, n, pad_left=True, pad_right=False, left_pad_symbol="<s>", right_pad_symbol=None)
    return list(n_grams)


def create_vocabulary(wordlist):
    vocabulary = {}
    keys = list(set(wordlist))
    for i, k in enumerate(keys):
        vocabulary[k] = i
    vocabulary["<s"] = len(keys)
    return vocabulary


def one_hot_transformer(vocabdict):
    one_hot_vector_dict = {}
    for k, v in vocabdict.items():
        one_hot_vector = [0] * len(vocabdict)
        one_hot_vector[v] = 1
        one_hot_vector_dict[k] = one_hot_vector
    return one_hot_vector_dict


def create_one_hot_representations(ngrams_, vector_dict):
    array = []
    for el in ngrams_:
        print(ngrams)
        vector = []
        for w in el[:-1]:
            vector += vector_dict[w]
        vector.append(el[-1])
        array.append(vector)
    return array


words = (get_tokens(args.inputfile))
vocabulary = create_vocabulary(words)
one_hot_vectors = one_hot_transformer(vocabulary)
n_grams = create_ngrams(words, 3)
#one_hot_matrix = create_one_hot_representations(n_grams, one_hot_vectors)

print(vocabulary)
print(one_hot_vectors)
print(n_grams)
#print(one_hot_matrix)




print("Loading data from file {}.".format(args.inputfile))
print("Starting from line {}.".format(args.startline))
if args.endline:
    print("Ending at line {}.".format(args.endline))
else:
    print("Ending at last line of file.")

print("Constructing {}-gram model.".format(args.ngram))
print("Writing table to {}.".format(args.outputfile))
    
# THERE ARE SOME CORNER CASES YOU HAVE TO DEAL WITH GIVEN THE INPUT
# PARAMETERS BY ANALYZING THE POSSIBLE ERROR CONDITIONS.
