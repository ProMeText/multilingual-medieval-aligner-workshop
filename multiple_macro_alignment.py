import json
import os

from numpyencoder import NumpyEncoder
import sys
import numpy as np
# import collatex
import graph_merge
import bertalign.utils as utils
import bertalign.syntactic_tokenization as syntactic_tokenization
from bertalign import Bertalign
import tests


result_a = [([0], [0]), ([1], [1]), ([2], [2]), ([3], [3]), ([4], [4]), ([5, 6, 7], [5, 6]), 
            ([8], [7, 8, 9, 10]), ([9], [11, 12]), ([10, 11], [13]), ([12], []), ([13], [14]), 
            ([14], [15, 16]), ([15], [17, 18]), ([16], [19]), ([17, 18, 19], [20, 21]), 
            ([20], [22]), ([21], [23, 24]), ([22], [25]), ([23], [26]), ([24], [27]), 
            ([25], [28, 29, 30]), ([26], [31, 32]), ([27, 28], [33, 34]), ([29], [35]), 
            ([30, 31], [36, 37]), ([32], [38]), ([33], [39]), ([34], []), ([35], []), 
            ([36], []), ([37], []), ([38], []), ([39], [])]

result_b = [([0], [0]), ([1, 2, 3], [1, 2]), ([4, 5], [3]), ([6, 7], [4]), ([8], [5]), 
            ([9, 10, 11], [6]), ([12], [7, 8]), ([13], [9, 10]), ([14], [11]),
            ([15], [12]), ([16, 17], [13, 14, 15]), ([18], [16]), ([19], [17]), 
            ([], [18]), ([20], [19, 20]), ([21], [21, 22]), ([22, 23, 24], [23]), 
            ([25, 26], [24, 25, 26]), ([27], [27]), ([28], [28, 29, 30]), ([29], [31]),
            ([30, 31, 32], [32]), ([33], [33]), ([34, 35], [34]), ([36, 37], [35]),
            ([38], [36]), ([39], [37, 38, 39])]

def create_pairs(full_list:list, main_wit_index:int) -> list:
    """
    From a list of witnesses and the main witness index, create all possible pairs with this witness.
    """
    pairs = []
    main_wit = full_list.pop(int(main_wit_index))
    for wit in full_list:
        pairs.append((main_wit, wit))
    print(pairs)
    return pairs


def blue_print(string):
    OKBLUE = '\033[94m'
    ENDC = '\033[0m'
    print(f"{OKBLUE}{string}{ENDC}")



def red_print(string):
    RED = '\033[31m'
    ENDC = '\033[0m'
    print(f"{RED}{string}{ENDC}")


class Aligner:
    def __init__(self, corpus_size:None, max_align=3, out_dir="default"):
        self.alignment_dict = dict()
        self.text_dict = dict()
        self.files_path = sys.argv[1:-2]
        self.main_file_index = sys.argv[-2]
        self.corpus_size = corpus_size
        self.max_align = max_align
        self.out_dir = out_dir

    def parallel_align(self):
        """
        This function procedes to the alignments two by two and then merges the alingments into one
        """
        pairs = create_pairs(self.files_path, self.main_file_index)
        for index, (main_wit, wit_to_compare) in enumerate(pairs):
            main_wit_name = main_wit.split("/")[-1].split(".")[0]
            wit_to_compare_name = wit_to_compare.split("/")[-1].split(".")[0]
            print(f"Aligning {main_wit} with {wit_to_compare}")
            first_tokenized_text = utils.clean_tokenized_content(syntactic_tokenization.syntactic_tokenization(main_wit, corpus_limit=self.corpus_size))
            print(len(first_tokenized_text))
            second_tokenized_text = utils.clean_tokenized_content(syntactic_tokenization.syntactic_tokenization(wit_to_compare, corpus_limit=self.corpus_size))
            try:
                os.mkdir(f"result_dir/{self.out_dir}/")
            except FileExistsError:
                pass
            utils.write_json(f"result_dir/{self.out_dir}/tokenized_{wit_to_compare_name}.json", first_tokenized_text)
            utils.write_json(f"result_dir/{self.out_dir}/tokenized_{wit_to_compare_name}.json", second_tokenized_text)
            utils.write_tokenized_text(f"result_dir/{self.out_dir}/tokenized_{wit_to_compare_name}.txt", second_tokenized_text)
            self.text_dict[0] = first_tokenized_text
            self.text_dict[index + 1] = second_tokenized_text
            aligner = Bertalign(first_tokenized_text, second_tokenized_text, max_align= self.max_align)
            aligner.align_sents()
            self.alignment_dict[index] = aligner.result
            utils.write_json(f"result_dir/{self.out_dir}/alignment_{str(index)}.json", aligner.result)
            utils.save_alignment_results(aligner.result, first_tokenized_text, second_tokenized_text,
                                         f"{main_wit_name}_{wit_to_compare_name}", out_dir)
        utils.write_json(f"result_dir/{self.out_dir}/alignment_dict.json", self.alignment_dict)

    def save_final_result(self, list_of_merged_alignments, MyAligner):
        """
        Saves result to tsv file
        """
        with open(f"result_dir/{self.out_dir}/final_result.tsv", "w") as output_text:
            output_text.write("\t".join(letter for letter in list_of_merged_alignments[0]) + "\n")
            # TODO: remplacer ça, c'est pas propre et ça sert à rien
            translation_table = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8}
            for alignment_unit in list_of_merged_alignments:
                output_text.write("|".join(value for value in alignment_unit['a']) + "\t")
                for index, witness in enumerate(list_of_merged_alignments[0]):
                    output_text.write("|".join(MyAligner.text_dict[translation_table[witness]][int(value)] for value in
                                               alignment_unit[witness]))
                    if index + 1 != len(list_of_merged_alignments[0]):
                        output_text.write("\t")
                output_text.write("\n")


if __name__ == '__main__':
    # Ce qui a été fait: le problème de l'alignement trivial (1 pour 1 dans tous les témoins) est réglé.
    # Des tests sont menés sur 1 pour plusieurs. 
    # Un test de graphe est mené pour voir si ça peut pas permettre de fusionner les lieux variants
    # Ça a l'air de marcher
    # TODO: augmenter la sensibilité à la différence sémantique pour apporter plus d'omissions dans le texte. La fin
    # Est beaucoup trop mal alignée, alors que ça irait bien avec + d'absence. 
    out_dir = sys.argv[-1]
    MyAligner = Aligner(corpus_size=300, max_align=3, out_dir=out_dir)
    MyAligner.parallel_align()
    utils.write_json(f"result_dir/{out_dir}/alignment_dict.json", MyAligner.alignment_dict)
    align_dict = utils.read_json(f"result_dir/{out_dir}/alignment_dict.json")
    list_of_merged_alignments = graph_merge.merge_alignment_table(align_dict)
    # On teste si on ne perd pas de noeuds textuels
    utils.test_tables_consistency(list_of_merged_alignments, 'abcde')
    MyAligner.save_final_result(list_of_merged_alignments, MyAligner)
    
    
                
            
    