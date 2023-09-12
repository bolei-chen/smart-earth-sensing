import sys
sys.path.append('/Users/lei/home/studyhall/smart-earth-sensing/lib') 
 
import numpy as np 
import pickle 
 
from scipy.fft import fft 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
from sklearn.metrics import silhouette_score 

from matplotlib import pyplot as plt 

from utils import div2chunks 
 
''' 
input:
    data_norm: a np array of shape (n, 1) consists of normal data samples
    data_abnorm: a np array of shape (n, 1) consists of abnormal data samples
    data_train: a np array of shape (n, 1) consists of training data samples
    sr: sampling rate
    pca_path: the path of a pickle file 
    normal_kmeans_dirpath: the path of a directory or a folder storing the kmeans models for normal data
    abnormal_kmeans_dirpath: the path of a directory or a folder storing the kmeans models for abnormal data
ouput:
    kmeans_norm: the kmeans model for normal data with highest silhouette score 
    kmeans_abnorm: the kmeans model for abnormal data with highest silhouette score 
''' 
def get_best_kmeans(data_norm, data_abnorm, data_train, sr, pca_path, normal_kmeans_dirpath, abnormal_kmeans_dirpath):
    fft_norm = raw2fft(data_norm, sr) 
    fft_abnorm = raw2fft(data_abnorm, sr) 
    fft_train = raw2fft(data_train, sr) 
    pca = get_pca(fft_train, 64, pca_path) 
    compressed_norm = pca.transform(fft_norm) 
    compressed_abnorm = pca.transform(fft_abnorm) 
    kmeansarr_norm = get_kmeans(compressed_norm, normal_kmeans_dirpath) 
    kmeansarr_abnorm = get_kmeans(compressed_abnorm, abnormal_kmeans_dirpath) 
    kmeans_norm = select_appropriate_kmeans(kmeansarr_norm, compressed_norm, normal_kmeans_dirpath) 
    kmeans_abnorm = select_appropriate_kmeans(kmeansarr_abnorm, compressed_abnorm, abnormal_kmeans_dirpath) 
    return kmeans_norm, kmeans_abnorm 

''' 
input:
    raw: a np array of shape (n, 1) 
    sr: sampling rate 
output:
    xs: a np array of shape (n / sr, sr / 2) consists of the fft value
''' 
def raw2fft(raw, sr):
    raw = np.array([s[0] for s in raw])
    raw = np.array(list(div2chunks(raw, sr))) 
    xs = np.array([fft(s)[:sr // 2].real for s in raw]) 
    return xs 

''' 
input: 
    xs: a np array of shape (n / sr, sr / 2) consists of the fft value
    n_components: the number of components for pca training 
    store_path: the path to store the pca model 
output:
    pca: the trained pca model 
''' 
def get_pca(xs, n_components, store_path):
    pca = PCA(n_components)
    pca.fit(xs) 
    with open(store_path, 'wb') as f:
        pickle.dump(pca, f)
    return pca 

''' 
input: 
    kmeansarr: an array of kmeans models 
output:
    a graph showing the sum of sqaured errors 
    the data point right at the elbow of the line indicates the best model 
'''
def elbow_test(kmeansarr):
    l = len(kmeansarr)
    ssh = [km.inertia_ for km in kmeansarr] 
    plt.plot(range(1, l + 1), ssh) 
    plt.ylabel("sum of square distance") 
    plt.xlabel("number of clusters") 
    plt.show() 
    return 0

''' 
input:
    xs: a np array of shape (n / sr, pca_n_components) consists of the pca compressed form of the raw data
    store_path: the folder path to store all the kmeans models 
output:
    kmeansarr: an array storing the trained kmeans models from n_clusters=2 to n_clusters=8
''' 
def get_kmeans(xs, store_path):
    kmeansarr = [] 
    for n_clusters in range(2, 9):
        kmeans = KMeans(n_clusters) 
        kmeans.fit(xs) 
        with open(store_path + '/k_means_' + str(n_clusters) + ".pkl", 'wb') as f:
            pickle.dump(kmeans, f) 
        kmeansarr.append(kmeans) 
    return kmeansarr 
     
''' 
input:
    kmeansarr: an array storing the trained kmeans models from n_clusters=2 to n_clusters=8
    xs: a np array of shape (n / sr, pca_n_components) consists of the pca compressed form of the raw data
    store_path: the folder path to store the kmeans models with the highest silhouette score 
output:
    model: the kmeans model with the best silhouette score 
''' 
def select_appropriate_kmeans(kmeansarr, xs, store_path):
    score2model = dict(zip([silhouette_score(xs, km.fit_predict(xs)) for km in kmeansarr], kmeansarr))
    max_score = max(score2model.keys())
    model = score2model[max_score] 
    with open(store_path + '/best_kmeans.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model 
