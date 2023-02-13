import ordering_finder
import artificial_data
import dyn_prog_improved_impl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from functools import lru_cache, wraps
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import pickle

def test_alg(data, order, sim, name, iters, freq_reports):
    """
    Calls the Algorithm with the given arguments and records the results
    """
    data = order(data)
    X, y = data.drop('target', axis=1), data['target']
    res_list, res_tree, start_cost, reports, final_cost_improvement = dyn_prog_improved_impl.main(X.to_numpy(), sim , iters, freq_reports)
    parent_dir = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(parent_dir, name)
    if not os.path.exists(path):
        os.mkdir(path)
    os.chdir(path)
    save_results(res_list, res_tree, name)
    save_plots(res_list, res_tree, name)
    reports_to_excel(start_cost, reports, final_cost_improvement, name, iters)

def save_results(res_list, res_tree, name):
    with open(f"{name}_list.pickle", 'wb') as f:
        pickle.dump(res_list, f)
    with open(f"{name}_tree.pickle", 'wb') as f:
        pickle.dump(res_tree, f)

def get_multiple_clusters(list_, n_inst):
    prev = 0
    class_ = [0 for _ in range(n_inst)]
    for num in range(len(list_)):
        print(prev)
        class_[prev:list_[num]] = [num for _ in range(list_[num]-prev)]
        prev = list_[num]
    class_[list_[-1]:] = [len(list_) for _ in range(n_inst - list_[-1])]
    return class_

def get_split_points(res_tree):
    splits = [res_tree.root.left.split, res_tree.root.split, res_tree.root.right.split]
    splits.sort()
    return splits

def save_plots(res_list, res_tree, name):
    """
    Saves plots of the found clustering.
    """
    fig, ax = plt.subplots()
    res_list = np.array(res_list)
    class_ = [0 if i>=res_tree.root.split else 1 for i in range(len(res_list))]
    ax.scatter(res_list[:,0], res_list[:,1], c=class_)
    fig.savefig(f"{name}.png")
    res_df = pd.DataFrame(res_list)
    
    fig, ax = plt.subplots()
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(res_df)
    ax.scatter(transformed[:,0], transformed[:,1], c=class_)
    fig.savefig(f"{name}_pca_scatter.png")
    
    fig, ax = plt.subplots()
    tsne = TSNE(n_components=2)
    X_reduced = tsne.fit_transform(res_df)
    ax.scatter(X_reduced[:,0], X_reduced[:,1], c=class_)
    fig.savefig(f"{name}_tSNE_scatter.png")

    fig, ax = plt.subplots()
    res_df["cluster"] = class_
    plot = pd.plotting.parallel_coordinates(
        res_df, 'cluster', color=('#556270', '#4ECDC4')
    )
    plot.figure.savefig(f"{name}_parallel_coordinates.png")

    splits = get_split_points(res_tree)
    class_ = get_multiple_clusters(splits, len(res_df))

    fig, ax = plt.subplots()
    ax.scatter(transformed[:,0], transformed[:,1], c=class_)
    fig.savefig(f"{name}_4cl_pca_scatter.png")
    
    fig, ax = plt.subplots()
    ax.scatter(X_reduced[:,0], X_reduced[:,1], c=class_)
    fig.savefig(f"{name}_4cl_tSNE_scatter.png")

    fig, ax = plt.subplots()
    res_df["cluster"] = class_
    plot = pd.plotting.parallel_coordinates(
        res_df, 'cluster', color=('#556270', '#4ECDC4')
    )
    plot.figure.savefig(f"{name}_4cl_parallel_coordinates.png")

def reports_to_excel(start_cost, reports, final_cost_improvement, sheet_name, iterations=100):
    """
    Saves the clustering results to xslx and csv files. 
    """
    rep_df = pd.DataFrame([[1, start_cost, "-"]], columns=["Iteration","Cost","Improvement"])
    rep_df = rep_df.append(pd.DataFrame(reports, columns=["Iteration","Cost","Improvement"]))
    rep_df = rep_df.append(pd.DataFrame([[iterations,final_cost_improvement[0], final_cost_improvement[1]]], columns=["Iteration","Cost","Improvement"]))
    rep_df.to_excel(f"{sheet_name}.xlsx",index=False)
    rep_df.to_csv(f"{sheet_name}.csv",index=False)

def np_cache(function):
    """wrapper for lru_cache to work with numpy arrays"""
    #source:
    @lru_cache(maxsize=None)
    def cached_wrapper(hashable_array, hashable_array2):
        array1 = np.array(hashable_array)
        array2 = np.array(hashable_array2)
        return function(array1, array2)

    @wraps(function)
    def wrapper(array1, array2):
        return cached_wrapper(tuple(array1), tuple(array2))

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper

def fetch_ucl_datasets():
    iris = load_iris()
    iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    iris_df['target'] = iris['target']
    wine = load_wine()
    wine_df = pd.DataFrame(wine['data'], columns=wine['feature_names'])
    wine_df['target'] = wine['target']
    breast_cancer = load_breast_cancer()
    breast_cancer_df = pd.DataFrame(breast_cancer['data'], columns=breast_cancer['feature_names'])
    breast_cancer_df['target'] = breast_cancer['target']
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
    glass_data = pd.read_csv(url, names=["Id","RI","Na","Mg","Al","Si","K","Ca","Ba","Fe","target"], header=None)
    glass_data = glass_data.drop(columns="Id")

    return iris_df, wine_df, breast_cancer_df, glass_data

@np_cache
def sim(a, b):
    return 1/(1+distance.euclidean(a, b))

def main(iters = 100, freq_reports = 1):
    """Fetch data and set parameters to test on."""
    iris_df, wine_df, breast_cancer_df, glass_data = fetch_ucl_datasets()
    noisy_circles, noisy_moons, blobs, aniso, varied = artificial_data.generate_datasets(300)

    orderings = {"PCA ordering": ordering_finder.pca_ordering}
    # "PCA ordering": ordering_finder.pca_ordering, "tSNE ordering": ordering_finder.tsne_ordering
    # "Greedy ordering": ordering_finder.greedy_ordering, "K-means ordering": ordering_finder.create_ordering
    # "Linkage ordering": ordering_finder.sch_ordering, "Random Ordering": ordering_finder.rand_ord
    datasets =  {"Iris Dataset": iris_df}
    #"Noisy Circles": noisy_circles, "Noisy Moons": noisy_moons, "Blobs": blobs, "Aniso": aniso, "Varied blobs": varied
    #"Iris Dataset": iris_df, "Wine Dataset": wine_df, "Glass Dataset": glass_data, "Breast Cancer Dataset": breast_cancer_df

    for order in orderings.keys():
        for dataset in datasets.keys():
            print(datasets[dataset])
            test_alg(datasets[dataset].copy(), orderings[order], sim, f"{dataset}, {order}, {iters} iters", iters, freq_reports)

if __name__ == "__main__":
    main()
