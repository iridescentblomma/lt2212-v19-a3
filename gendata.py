import os, sys
import re
import random
import argparse
import numpy as np
import pandas as pd
from nltk import ngrams
import linecache

# gendata.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here. You may not use the
# scikit-learn OneHotEncoder, or any related automatic one-hot encoders.

parser = argparse.ArgumentParser(description="Convert text to features")
parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int, default=3,
                    help="The length of ngram to be considered (default 3).")
parser.add_argument("-S", "--start", metavar="S", dest="startline", type=int,
                    default=0,
                    help="What line of the input data file to start from. Default is 0, the first line.")
parser.add_argument("-E", "--end", metavar="E", dest="endline",
                    type=int, default=None,
                    help="What line of the input data file to end on. Default is None, whatever the last line is.")
parser.add_argument("-T", "--test", metavar="T", dest="testdata", type=int,
                    help="The numbers of lines designated as the testdata.")
parser.add_argument("inputfile", type=str,
                    help="The file name containing the text data.")
parser.add_argument("outputfile", type=str,
                    help="The name of the output file for the feature table.")
parser.add_argument("-P", "--POS", dest="POS", default=False, action="store_true",
                    help="Create vector representations with POS with POS as label.")

args = parser.parse_args()


def divide_input(file, start, end, nt):
    text = []
    if end is None:
        end = 1000000
    for i in range(start, end):
        li = linecache.getline(file, i)
        line = re.sub(r"\n", "", li)
        text.append(line)
    all_lines = " ".join(text)
    randoms = []
    for r in range(nt):
        randoms.append(random.randint(0, len(text)))
    test_li = []
    train_li = []
    for l,x in enumerate(text):
        if l in randoms:
            test_li.append(x)
        else:
            train_li.append(x)
    test_lines = " ".join(test_li)
    train_lines = " ".join(train_li)
    return all_lines, test_lines, train_lines


def get_list_of_tokens(string, pos):
    raw_tokens = string.split(" ")
    tokens_POS_li = []
    for t in raw_tokens:
        tokens_POS_li.append((t.split("/")))
    for p in tokens_POS_li:
        if p[0] == "":
            tokens_POS_li.remove(p)
    if pos:
        tokens_POS_tu = []
        for tp in tokens_POS_li:
            tokens_POS_tu.append(tuple(tp))
        return tokens_POS_tu
    else:
        tokens = []
        for t_p in tokens_POS_li:
            tokens.append(t_p[0])
        return tokens


def create_vocabulary(wordlist):
    vocabulary = {}
    keys = list(set(wordlist))
    for i, k in enumerate(keys):
        vocabulary[k] = i
    vocabulary["<s>"] = len(keys)
    return vocabulary


def one_hot_transformer(vocabdict):
    one_hot_vector_dict = {}
    for k, v in vocabdict.items():
        one_hot_vector = [0] * len(vocabdict)
        one_hot_vector[v] = 1
        one_hot_vector_dict[k] = one_hot_vector
    return one_hot_vector_dict


def create_ngrams(tokens, n):
    n_grams = ngrams(tokens, n, pad_left=True, pad_right=False, left_pad_symbol="<s>", right_pad_symbol=None)
    return list(n_grams)


def create_one_hot_representations(ngrams_, vector_dict, pos):
    all_vectors = []
    for el in ngrams_:
        print(el)
        vector = []
        for w in el[:-1]:
            vector += vector_dict[w]
        if pos:
            vector.append(el[-1][1])
        else:
            vector.append(el[-1])
        all_vectors.append(vector)
    representation = np.array(all_vectors)
    return representation


if args.startline > args.endline and args.testdata > (args.endline - args.startline):
    print("Starting line has to be smaller than ending line!",
          "Number of lines desired in test must be smaller than selected lines through -S and -E")
elif args.startline > args.endline:
    print("Starting line has to be smaller than ending line!")
elif args.testdata > (args.endline - args.startline):
    print("Number of lines desired in test must be smaller than selected lines through -S and -E")
else:
    all_lines, test_lines, train_lines = divide_input(args.inputfile, args.startline, args.endline, args.testdata)


corpus = get_list_of_tokens(all_lines, args.POS)
test = get_list_of_tokens(test_lines, args.POS)
train = get_list_of_tokens(train_lines, args.POS)


if args.ngram < 2:
    print("Ngrams must at least be trigrams!")
else:
    n_grams_test = create_ngrams(test, args.ngram)
    n_grams_train = create_ngrams(train, args.ngram)

vocabulary = create_vocabulary(corpus)
one_hot_vectors = one_hot_transformer(vocabulary)


repr_test = pd.DataFrame(create_one_hot_representations(n_grams_test, one_hot_vectors, args.POS))
repr_train = pd.DataFrame(create_one_hot_representations(n_grams_train, one_hot_vectors, args.POS))



#np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)
pd.set_option("display.width", 10**1000)


repr_test.to_csv(args.outputfile+".test.csv", header=None, index=None)
repr_train.to_csv(args.outputfile+".train.csv", header=None, index=None)


print("Loading data from file {}.".format(args.inputfile))
print("Starting from line {}.".format(args.startline))
if args.endline:
    print("Ending at line {}.".format(args.endline))
else:
    print("Ending at last line of file.")

print("Constructing {}-gram model.".format(args.ngram))
print("Writing table to {}.".format(args.outputfile))

