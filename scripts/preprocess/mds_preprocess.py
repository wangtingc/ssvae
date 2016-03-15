"""
This script is what created the dataset pickled.

1) You need to download this file and put it in the same directory as this file.
https://github.com/moses-smt/mosesdecoder/raw/master/scripts/tokenizer/tokenizer.perl . Give it execution permission.

2) Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/ and extract it in the current directory.

3) Then run this script.
"""

dataset_path='/Tmp/bastienf/aclImdb/'

import numpy as np
import cPickle as pkl
import gzip

from collections import OrderedDict
import nltk.data

import glob
import os

from subprocess import Popen, PIPE

# tokenizer.perl is from Moses: https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer
tokenizer_cmd = ['/home/wead/codes/cpp/mosesdecoder/scripts/tokenizer/tokenizer.perl', '-l', 'en', '-q', '-']


def tokenize(sentences):

    print 'Tokenizing..',
    text = "\n".join(sentences)
    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
    tok_text, _ = tokenizer.communicate(text)
    toks = tok_text.split('\n')[:-1]
    print 'Done'

    return toks


def filt_line(line):
    return line.strip().replace('<br />', ' ')


def build_dict(path, include_unlabel):
    sentences = []
    currdir = os.getcwd()

    dataset = pkl.load(open(path))
    text_l = dataset['text_l']
    text_u = dataset['text_u']

    sentences = text_l + text_u
    os.chdir(currdir)

    sentences = tokenize(sentences)

    print 'Building dictionary..',
    wordcount = dict()
    for ss in sentences:
        words = ss.strip().lower().split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    counts = wordcount.values()
    keys = wordcount.keys()

    sorted_idx = np.argsort(counts)[::-1]

    worddict = dict()

    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)

    print np.sum(counts), ' total words ', len(keys), ' unique words'

    return worddict


def grab_data(text, dictionary):
    sent_tokenize = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = []
    currdir = os.getcwd()

    sent_divide = [0]
    for p in text:
        line = filt_line(p.decode('utf8'))
        sentences = sent_tokenize.tokenize(line)
        sentences = [s.encode('utf8') for s in sentences]
        sent_divide.append(sent_divide[-1] + len(sentences))
        sents.extend(sentences)

    sents = tokenize(sents)

    reviews = [None] * (len(sent_divide) - 1)
    for idx in xrange(len(sent_divide) - 1):
        seqs = []
        start = sent_divide[idx]
        end = sent_divide[idx+1]
        for sentence in sents[start: end]:
            words = sentence.strip().lower().split()
            seqs.append([dictionary[w] if w in dictionary else 1 for w in words])
        if idx % 1000 == 0:
            print idx
        reviews[idx] = seqs

    return reviews


def sort_by_len(reviews, labels):
    def len_argsort(seq):
        def v(x):
            r = seq[x]
            n = 0
            for i in r:
                n += len(i)
            return n
        return sorted(range(len(seq)), key=lambda x: v(x))

    sorted_idx = len_argsort(reviews)
    reviews = [reviews[i] for i in sorted_idx]
    labels = [labels[i] for i in sorted_idx]

    return reviews, labels


def split_dataset(text, label):
    category = []
    category.append([[], [], [], []])
    category.append([[], [], [], []])

    for i in range(len(label)):
        idx0 = label[i][0]
        idx1 = label[i][1]
        category[idx0][idx1].append(text[i])

    for i in range(2):
        for j in range(4):
            print len(category[i][j])

    train = []
    train.append([[],[],[],[]])
    train.append([[],[],[],[]])
    test = []
    test.append([[],[],[],[]])
    test.append([[],[],[],[]])


    n_train_per_label = len(category[i][j]) * 0.8
    n_test_per_label = len(category[i][j]) * 0.2

    for i in range(2):
        for j in range(4):
            for k in range(len(category[i][j])/5):
                train[i][j].extend(category[i][j][5*k: 5*k+4])
                test[i][j].append(category[i][j][5*k+4])

    train_x = []
    train_y = []
    for i in range(2):
        for j in range(4):
            #print len(category[i][j])
            #print category[i][j][0]
            y_per_label = [[i, j]] * len(train[i][j])
            x_per_label = train[i][j]
            train_x.extend(x_per_label)
            train_y.extend(y_per_label)

    test_x = []
    test_y = []
    for i in range(2):
        for j in range(4):
            y_per_label = [[i, j]] * len(test[i][j])
            x_per_label = test[i][j]
            test_x.extend(x_per_label)
            test_y.extend(y_per_label)

    return (train_x, train_y), (test_x, test_y)

def categorize(binary):
    y = []
    y0_binary = np.asarray(binary)[:, :2]
    y1_binary = np.asarray(binary)[:, 2:]

    y0 = np.argmax(y0_binary, axis=1)
    y1 = np.argmax(y1_binary, axis=1)

    y = np.asarray([y0, y1])
    y = y.T

    print y.shape

    return y.tolist()


def main():
    # Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/
    path = '/workspace/datasets/multi-domain-sentiment/rawTextLabel.pkl'
    dictionary = build_dict(path, True)

    raw = pkl.load(open(path))
    label_l = categorize(raw['label_l'])
    label_u = categorize(raw['label_u'])
    label_u = [[-1, label_u[i][1]] for i in range(len(label_u))]

    text_l = grab_data(raw['text_l'], dictionary)
    text_l, label_l = sort_by_len(text_l, label_l)
    train, test = split_dataset(text_l, label_l)
    train_x, train_y = train
    test_x, test_y = test

    text_u = grab_data(raw['text_u'], dictionary)
    text_u, label_u = sort_by_len(text_u, label_u)
    unlabel_x = text_u
    unlabel_y = label_u

    f_name = '../../data/proc/mds/mds.pkl.gz'
    f = gzip.open(f_name, 'wb')
    pkl.dump((train_x, train_y), f, -1)
    pkl.dump((test_x, test_y), f, -1)
    pkl.dump((unlabel_x, unlabel_y), f, -1)
    f.close()

    f_name = '../../data/proc/mds/mds.dict.pkl.gz'
    f = gzip.open(f_name, 'wb')
    pkl.dump(dictionary, f, -1)
    f.close()

if __name__ == '__main__':
    main()
