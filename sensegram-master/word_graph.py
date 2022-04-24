import faiss
import codecs
from time import time
from gensim.models import KeyedVectors
import numpy as np


def compute_graph_of_related_words(vectors_fpath, neighbours_fpath, neighbors=200):
    print("Start collection of word neighbours.")
    tic = time()
    index, w2v = build_vector_index(vectors_fpath)
    compute_neighbours(index, w2v, neighbours_fpath, neighbors)
    print("Elapsed: {:f} sec.".format(time() - tic))


def build_vector_index(w2v_fpath):
    w2v = KeyedVectors.load_word2vec_format(w2v_fpath, binary=False, unicode_errors='ignore')
    # print("fried",w2v.get_vector('fried'))
    # print("the",w2v.get_vector('the'))
    # print("w2v before-----",w2v.vectors)
    # all_normed_vectors = w2v.get_normed_vectors()
    # # w2v.init_sims(replace=True)
    # print("w2v after-----",all_normed_vectors)
    # index = faiss.IndexFlatIP(w2v.vector_size)

    # index.add(all_normed_vectors)
    vectors = w2v.vectors
    num_vectors = len(vectors)
    vector_dim = w2v.vector_size
    # vectors = np.random.rand(num_vectors, vector_dim)
    # print(w2v.vectors.dtype)
    # print('00000000000000',type(w2v.vectors))
    
    quantizer = faiss.IndexFlatIP(vector_dim)
    index = faiss.IndexIVFFlat(quantizer, vector_dim, int(np.sqrt(num_vectors)), faiss.METRIC_INNER_PRODUCT)
    train_vectors = vectors[:int(num_vectors/2)].copy()
    faiss.normalize_L2(train_vectors)
    index.train(train_vectors)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    #gensim.models.keyedvectors.KeyedVectors.fill_norms()


    return index, w2v


def compute_neighbours(index, w2v, nns_fpath, neighbors=5):
    tic = time()
    with codecs.open(nns_fpath, "w", "utf-8") as output:
        X = w2v.vectors
        D, I = index.search(X, neighbors + 1)

        j = 0
        for _D, _I in zip(D, I):
            for n, (d, i) in enumerate(zip(_D.ravel(), _I.ravel())):
                if n > 0 and d>0:
                    output.write("{}\t{}\t{:f}\n".format(w2v.index_to_key[j], w2v.index_to_key[i], d))
            j += 1

        print("Word graph:", nns_fpath)
        print("Elapsed: {:f} sec.".format(time() - tic))


