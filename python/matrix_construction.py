import re
import sys
import pandas
import bertalign.utils as utils
import numpy as np




def compute_presence_absence(results):
    absences = 0
    for alignment_item in results:
        wit_a, wit_b = alignment_item
        if wit_a == [] or wit_b == []:
            absences += 1
    return absences

def tokenize(texts):
    tokenized = {}
    import bertalign.syntactic_tokenization as syntactic_tokenization
    for text in texts:
        tokenized[text.split("/")[-1].split(".")[0]] = utils.clean_tokenized_content(syntactic_tokenization.syntactic_tokenization(text))
    return tokenized

def parallel_align(tokenized_texts:dict):
    from bertalign import Bertalign
    possible_pairs = utils.construct_pairs(list(tokenized_texts))
    print("Possible pairs:")
    print(possible_pairs)
    presence_absence_results = {}
    for text_a, text_b in possible_pairs:
        print(f"Comparing {text_a} and {text_b}.")
        aligner_1 = Bertalign(tokenized[text_a], tokenized[text_b])
        aligner_1.align_sents()
        utils.save_alignment_results(aligner_1.result, tokenized[text_a], tokenized[text_b], f"{text_a}_{text_b}")
        absence_results = compute_presence_absence(aligner_1.result)
        # We create a dictionnary with the results for each pair
        try:
            presence_absence_results[text_a][text_b] = absence_results
        except KeyError:
            presence_absence_results[text_a] = {}
            presence_absence_results[text_a][text_b] = absence_results
        try:
            presence_absence_results[text_b][text_a] = absence_results
        except KeyError:
            presence_absence_results[text_b] = {}
            presence_absence_results[text_b][text_a] = absence_results
    return presence_absence_results


def presence_absence_to_matrix(absence_results: dict):
    # https://stackoverflow.com/a/10628728
    print(absence_results)
    df = pandas.DataFrame(absence_results).T.fillna(0)

    # Let's sort the data to make the matrix symetric
    sort_idx = df.sort_index(axis=0)
    sort_idx = sort_idx.sort_index(axis=1)
    print(sort_idx)
    with open("result_dir/absence_matrix.csv", "w") as matrix:
        matrix.write(sort_idx.to_csv())


def blue_print(string):
    OKBLUE = '\033[94m'
    ENDC = '\033[0m'
    print(f"{OKBLUE}{string}{ENDC}")


def red_print(string):
    RED = '\033[31m'
    ENDC = '\033[0m'
    print(f"{RED}{string}{ENDC}")


if __name__ == '__main__':
    # Ce qui a été fait: le problème de l'alignement trivial (1 pour 1 dans tous les témoins) est réglé.
    # Des tests sont menés sur 1 pour plusieurs. 
    # Un test de graphe est mené pour voir si ça peut pas permettre de fusionner les lieux variants
    # Ça a l'air de marcher
    # TODO: augmenter la sensibilité à la différence sémantique pour apporter plus d'omissions dans le texte. La fin
    # Est beaucoup trop mal alignée, alors que ça irait bien avec + d'absence. 
    texts = sys.argv[1:]
    tokenized = tokenize(texts)
    absence_results = parallel_align(tokenized_texts=tokenized)
    presence_absence_to_matrix(absence_results)
