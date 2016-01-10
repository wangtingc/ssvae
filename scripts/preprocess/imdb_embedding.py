import cPickle as pkl
import gzip
import numpy as np

def load_pre_imdb(filename):
    f = open(filename, 'r')
    train = pkl.load(f)
    dev = pkl.load(f)
    test = pkl.load(f)
    voc = pkl.load(f)
    f.close()

    return (train, dev, test, voc)


def load_glove(filename):
    glove = {}
    f = gzip.open(filename, 'r')
    for line in f:
        key = line.split()[0]
        val = [float(i) for i in line.split()[1:]]
        glove[key] = val

    return glove


def build_emb(word_dict, word_emb):
    # the max value(indice) is the length for embedding matrix - 1
    n_words = 0
    for i in word_dict.values():
        n_words = i if n_words < i else n_words
    n_words += 1
    dim_emb = len(word_emb[word_emb.keys()[0]])
    W_emb = np.random.rand(n_words, dim_emb) * 0.01
    word_emb_keys = set(word_emb.keys())
    for i in word_dict.keys():
        if i in word_emb_keys:
            W_emb[word_dict[i]] = word_emb[i]

    return W_emb


if __name__ == '__main__':
    imdb_filename = '../../data/proc/imdb_l.dict.pkl.gz'
    glove_filename = '../../data/raw/glove/glove.6B.100d.txt.gz'

    #imdb = load_pre_imdb(imdb_filename)
    imdb_dict = pkl.load(gzip.open(imdb_filename))
    glove = load_glove(glove_filename)

    words_imdb = set(imdb_dict.keys())
    words_glove = set(glove.keys())
    n_words_imdb = len(words_imdb)
    n_words_glove = len(words_glove)

    print('imdb_filename', imdb_filename)
    print('glove_filename', glove_filename)

    print('n_words_imbd', n_words_imdb)
    print('n_words_glove', n_words_glove)

    # overlapping
    '''
    for i in xrange(10):
        n_words_imdb_sub = int(n_words_imdb * (i+1) * 1.0 / 100)
        words_imdb_sub = set(dict(words_imdb[: n_words_imdb_sub]).keys())
        overlapping = len(words_imdb_sub & words_glove)* 1.0 / n_words_imdb_sub
        print words_imdb_sub - words_glove
        #print('Overlapping first %d words %f: %f'%( n_words_imdb_sub, (i+1) * 0.01, overlapping))
    '''
    print('The percentage of imdb vocabulary covered by glove is', len(words_imdb & words_glove) * 1.0 / len(words_imdb))

    W_emb = build_emb(imdb_dict, glove)
    W_emb_filename = '../../data/proc/imdb_emb.pkl.gz'

    with gzip.open(W_emb_filename, 'wb') as f:
        pkl.dump(W_emb, f)
