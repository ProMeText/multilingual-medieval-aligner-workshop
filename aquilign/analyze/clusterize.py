import bertalign.utils as utils
from sklearn import preprocessing
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import sys

def create_clusters(path):
    """
    This function converts the distance matrix into a cluster matrix, using multiple methods.
    It take a json file containing all the distance pairs, turns it into a matrix, and then performs the
    clusterisation
    """
    print(path)
    np.set_printoptions(threshold=np.inf, suppress=True, linewidth=np.inf)
    alignements = utils.read_json(path)
    print(len(alignements))
    for index, alignement in enumerate(alignements):
        print(f"Unit {index}")
        if alignement == []:
            continue
        elif alignement == {}:
            continue
        similarities = [(wits, similarities) for wits, similarities in alignement.items()]
        # https://stackoverflow.com/a/16193637
        similarities.sort(key=lambda x: (int(x[0].split("-")[0]), int(x[0].split("-")[1])))
        out_list = []
        range_of_elements_a = list(set([element[0].split("-")[0] for element in similarities]))
        range_of_elements_b = list(set([element[0].split("-")[1] for element in similarities]))
        full_list = list(set(range_of_elements_a + range_of_elements_b))
        full_list.sort()
        for i in full_list:
            interm_list = []
            for j in full_list:
                if i == j:
                    interm_list.append(1)
                else:
                    interm_list.append([elem[1] for elem in similarities if elem[0] == f"{i}-{j}" or elem[0] == f"{j}-{i}"][0])
            out_list.append(interm_list)
        similarities_as_array = np.asarray(out_list)
        model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.45, linkage='complete').fit(similarities_as_array)
        model_2 = DBSCAN(min_samples=2).fit(similarities_as_array)
        print(model.labels_)
        print(model_2.labels_)

if __name__ == '__main__':
    create_clusters(sys.argv[1])