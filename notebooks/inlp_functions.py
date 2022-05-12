import sys
sys.path.append("../src")
sys.path.append("../data/embeddings")
from gensim.models.keyedvectors import Word2VecKeyedVectors
from gensim.models import KeyedVectors
import numpy as np
import sklearn
from sklearn.manifold import TSNE
import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['agg.path.chunksize'] = 10000

def tsne(vecs, labels, title="", ind2label=None, words=None, metric="l2"):
    tsne = TSNE(n_components=2)  # , angle = 0.5, perplexity = 20)
    vecs_2d = tsne.fit_transform(vecs)
    label_names = sorted(list(set(labels.tolist())))
    num_labels = len(label_names)

    names = sorted(set(labels.tolist()))

    plt.figure(figsize=(6, 5))
    colors = "red", "blue"
    for i, c, label in zip(sorted(set(labels.tolist())), colors, names):
        plt.scatter(vecs_2d[labels == i, 0], vecs_2d[labels == i, 1], c=c,
                    label=label if ind2label is None else ind2label[label], alpha=0.3, marker="s" if i == 0 else "o")
        plt.legend(loc="upper right")

    plt.title(title)
    plt.savefig("embeddings.{}.png".format(title), dpi=600)
    plt.show()
    return vecs_2d

def load_word_vectors(fname):
    model = KeyedVectors.load_word2vec_format(fname, binary=False)
    vecs = model.vectors
    words = list(model.index_to_key)
    return model, vecs, words


def project_on_gender_subspaces(gender_vector, model: Word2VecKeyedVectors, n=2500):
    group1 = model.similar_by_vector(gender_vector, topn=n, restrict_vocab=None)
    group2 = model.similar_by_vector(-gender_vector, topn=n, restrict_vocab=None)

    all_sims = model.similar_by_vector(gender_vector, topn=len(model.vectors), restrict_vocab=None)
    eps = 0.03
    idx = [i for i in range(len(all_sims)) if abs(all_sims[i][1]) < eps]
    samp = set(np.random.choice(idx, size=n))
    neut = [s for i, s in enumerate(all_sims) if i in samp]
    return group1, group2, neut


def get_vectors(word_list: list, model: Word2VecKeyedVectors):
    vecs = []
    for w in word_list:
        vecs.append(model[w])

    vecs = np.array(vecs)

    return vecs


def get_bias_by_neighbors(model, v, gender_direction, topn):
    neighbors = model.similar_by_vector(v, topn=topn)
    neighbors_words = [n for n, _ in neighbors]

    # bias = len([n for n in neighbors_words if n in gendered_words])
    bias = len([n for n in neighbors_words if model.cosine_similarities(model[n], [gender_direction])[0] > 0])
    bias /= (1. * topn)
    return bias


def save_in_word2vec_format(vecs: np.ndarray, words: np.ndarray, fname: str):
    with open(fname, "w", encoding="utf-8") as f:
        f.write(str(len(vecs)) + " " + "300" + "\n")
        for i, (v, w) in tqdm.tqdm_notebook(enumerate(zip(vecs, words))):
            vec_as_str = " ".join([str(x) for x in v])
            f.write(w + " " + vec_as_str + "\n")

def perform_purity_test(vecs, k, labels_true):
    np.random.seed(0)
    clustering = sklearn.cluster.KMeans(n_clusters=k)
    clustering.fit(vecs)
    labels_pred = clustering.labels_
    score = sklearn.metrics.homogeneity_score(labels_true, labels_pred)
    return score


def compute_v_measure(vecs, labels_true, k=2):
    np.random.seed(0)
    clustering = sklearn.cluster.KMeans(n_clusters=k)
    clustering.fit(vecs)
    labels_pred = clustering.labels_
    return sklearn.metrics.v_measure_score(labels_true, labels_pred)