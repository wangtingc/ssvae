"""
This script is what created the dataset pickled.

1) You need to download this file and put it in the same directory as this file.
https://github.com/moses-smt/mosesdecoder/raw/master/scripts/tokenizer/tokenizer.perl . Give it execution permission.

2) Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/ and extract it in the current directory.

3) Then run this script.
"""

dataset_path='/Tmp/bastienf/aclImdb/'

import numpy
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
    os.chdir('%s/pos/' % path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(filt_line(f.readline()))
    os.chdir('%s/neg/' % path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(filt_line(f.readline()))

    if include_unlabel:
        os.chdir('%s/unsup/' % path)
        for ff in glob.glob("*.txt"):
            with open(ff, 'r') as f:
                sentences.append(filt_line(f.readline()))

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

    sorted_idx = numpy.argsort(counts)[::-1]

    worddict = dict()

    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)

    print numpy.sum(counts), ' total words ', len(keys), ' unique words'

    return worddict


def grab_data(path, dictionary):
    sent_tokenize = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = []
    currdir = os.getcwd()
    os.chdir(path)

    cnt_sent = 0
    sent_divide = [0]
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            line = filt_line(f.readline().decode('utf8'))
            sentences = sent_tokenize.tokenize(line)
            sentences = [s.encode('utf8') for s in sentences]
            cnt_sent += len(sentences)
            sent_divide.append(cnt_sent)
            sents.extend(sentences)

    os.chdir(currdir)
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

    return reviews


def main():
    data_type = 'u_sd' #('l', 'u', 'u_sd')
    print('Building data with', data_type)

    # Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/
    path = '/home/wead/datasets/aclImdb/'
    if data_type == 'l' or data_type == 'u_sd':
        dictionary = build_dict(os.path.join(path, 'train'), False)
    else:
        dictionary = build_dict(os.path.join(path, 'train'), True)

    train_x_pos = grab_data(path+'train/pos', dictionary)
    train_x_neg = grab_data(path+'train/neg', dictionary)
    if data_type == 'u' or data_type == 'u_sd':
        unlabel_x = grab_data(path+'train/unsup', dictionary)
    else:
        unlabel_x = []

    train_x = train_x_pos + train_x_neg
    train_y = [1] * len(train_x_pos) + [0] * len(train_x_neg)

    test_x_pos = grab_data(path+'test/pos', dictionary)
    test_x_neg = grab_data(path+'test/neg', dictionary)
    test_x = test_x_pos + test_x_neg
    test_y = [1] * len(test_x_pos) + [0] * len(test_x_neg)

    f_name = '../../data/proc/imdb_%s.pkl.gz' % data_type
    f = gzip.open(f_name, 'wb')
    pkl.dump((train_x, train_y), f, -1)
    pkl.dump((test_x, test_y), f, -1)
    pkl.dump(unlabel_x, f, -1)
    f.close()

    f_name = '../../data/proc/imdb_%s.dict.pkl.gz' % data_type
    f = gzip.open(f_name, 'wb')
    pkl.dump(dictionary, f, -1)
    f.close()

if __name__ == '__main__':
    main()
