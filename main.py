import json
import os

import string
from numpyencoder import NumpyEncoder
import sys
import numpy as np
# import collatex
import graph_merge
import bertalign.utils as utils
import bertalign.syntactic_tokenization as syntactic_tokenization
from bertalign.Bertalign import Bertalign
import pandas as pd

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
    """
    La classe Aligner initialise le moteur d'alignement, fondé sur Bertalign
    """
    def __init__(self, corpus_size:None, max_align=3, out_dir="default",use_punctuation=True):
        self.alignment_dict = dict()
        self.text_dict = dict()
        self.files_path = sys.argv[1:-3]
        self.main_file_index = sys.argv[-3]
        self.corpus_size = corpus_size
        self.max_align = max_align
        self.out_dir = out_dir
        self.use_punctiation = use_punctuation
        try:
            os.mkdir(f"result_dir/{self.out_dir}/")
        except FileExistsError:
            pass
        
        # Let's check the paths are correct
        for file in self.files_path:
            assert os.path.isfile(file), f"Vérifier le chemin: {file}"
            
        assert self.main_file_index.isdigit(), "L'avant-dernier paramètre doit être un nombre"

    def parallel_align(self):
        """
        This function procedes to the alignments two by two and then merges the alignments into a single alignement
        """
        pairs = create_pairs(self.files_path, self.main_file_index)
        first_tokenized_text = utils.clean_tokenized_content(syntactic_tokenization.syntactic_tokenization(pairs[0][0], corpus_limit=self.corpus_size, use_punctuation=True))
        assert first_tokenized_text != [], "Erreur avec le texte tokénisé du témoin base"
        
        main_wit_name = pairs[0][0].split("/")[-1].split(".")[0]
        utils.write_json(f"result_dir/{self.out_dir}/tokenized_{main_wit_name}.json", first_tokenized_text)
        utils.write_tokenized_text(f"result_dir/{self.out_dir}/tokenized_{main_wit_name}.txt", first_tokenized_text)
        
        # Let's loop and align each pair
        for index, (main_wit, wit_to_compare) in enumerate(pairs):
            main_wit_name = main_wit.split("/")[-1].split(".")[0]
            wit_to_compare_name = wit_to_compare.split("/")[-1].split(".")[0]
            print(f"Aligning {main_wit} with {wit_to_compare}")
            print(len(first_tokenized_text))
            second_tokenized_text = utils.clean_tokenized_content(syntactic_tokenization.syntactic_tokenization(wit_to_compare, corpus_limit=self.corpus_size, use_punctuation=True))
            assert second_tokenized_text != [], f"Erreur avec le texte tokénisé du témoin comparé {wit_to_compare_name}"
            utils.write_json(f"result_dir/{self.out_dir}/tokenized_{wit_to_compare_name}.json", second_tokenized_text)
            utils.write_tokenized_text(f"result_dir/{self.out_dir}/tokenized_{wit_to_compare_name}.txt", second_tokenized_text)
            
            # This dict will be used to create the alignment table in csv format
            self.text_dict[0] = first_tokenized_text
            self.text_dict[index + 1] = second_tokenized_text
            
            # Let's align the texts
            
            
            # Tests de profil et de paramètres
            profile = 0
            if profile == 0:
                margin = True
                len_penality = True
            else:
                margin = False
                len_penality = True
            aligner = Bertalign(first_tokenized_text, second_tokenized_text, max_align= self.max_align, win=5, skip=-.2, margin=margin, len_penalty=len_penality)
            aligner.align_sents()
            
            # We append the result to the alignment dictionnary
            self.alignment_dict[index] = aligner.result
            utils.write_json(f"result_dir/{self.out_dir}/alignment_{str(index)}.json", aligner.result)
            utils.save_alignment_results(aligner.result, first_tokenized_text, second_tokenized_text,
                                         f"{main_wit_name}_{wit_to_compare_name}", self.out_dir)
        utils.write_json(f"result_dir/{self.out_dir}/alignment_dict.json", self.alignment_dict)

    def save_final_result(self, merged_alignments:list, file_titles:list):
        """
        Saves result to csv file
        """
        filenames = [path.split("/")[-1] for path in file_titles]
        with open(f"result_dir/{self.out_dir}/final_result.csv", "w") as output_text:
            output_text.write("," + ",".join(filenames) + "\n")
            # TODO: remplacer ça, c'est pas propre et ça sert à rien
            translation_table = {letter:index for index, letter in enumerate(string.ascii_lowercase)}
            for alignment_unit in merged_alignments:
                output_text.write("|".join(value for value in alignment_unit['a']) + ",")
                for index, witness in enumerate(merged_alignments[0]):
                    output_text.write("|".join(self.text_dict[translation_table[witness]][int(value)] for value in
                                               alignment_unit[witness]))
                    if index + 1 != len(merged_alignments[0]):
                        output_text.write(",")
                output_text.write("\n")
        
        
        with open(f"result_dir/{self.out_dir}/readable.csv", "w") as output_text:
            output_text.write(",".join(filenames) + "\n")
            # TODO: remplacer ça, c'est pas propre et ça sert à rien
            translation_table = {letter:index for index, letter in enumerate(string.ascii_lowercase)}
            for alignment_unit in merged_alignments:
                for index, witness in enumerate(merged_alignments[0]):
                    output_text.write(" ".join(self.text_dict[translation_table[witness]][int(value)] for value in
                                               alignment_unit[witness]))
                    if index + 1 != len(merged_alignments[0]):
                        output_text.write(",")
                output_text.write("\n")
        
        with open(f"result_dir/{self.out_dir}/final_result_as_index.csv", "w") as output_text:
            output_text.write("," + ",".join(filenames) + "\n")
            for alignment_unit in merged_alignments:
                for index, witness in enumerate(merged_alignments[0]):
                    output_text.write("|".join(value for value in
                                               alignment_unit[witness]))
                    if index + 1 != len(merged_alignments[0]):
                        output_text.write(",")
                output_text.write("\n")

        data = pd.read_csv(f"result_dir/{self.out_dir}/final_result.csv")
        # Convert the DataFrame to an HTML table
        html_table = data.to_html()
        full_html_file = f"""<html>
                          <head>
                          <title>Alignement final</title>
                            <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
                            </head>
                          <body>
                          {html_table}
                          </body>
                    </html>"""
        with open(f"result_dir/{self.out_dir}/final_result.html", "w") as output_html:
            output_html.write(full_html_file)


def run_alignments():
    # TODO: augmenter la sensibilité à la différence sémantique pour apporter plus d'omissions dans le texte. La fin
    # Est beaucoup trop mal alignée, alors que ça irait bien avec + d'absence. Ça doit être possible vu que des omissions sont créés.
    out_dir = sys.argv[-2]
    use_punctuation = bool(sys.argv[-1])
    print(f"Punctuation for tokenization: {use_punctuation}")
    MyAligner = Aligner(corpus_size=None, max_align=3, out_dir=out_dir, use_punctuation=use_punctuation)
    MyAligner.parallel_align()
    utils.write_json(f"result_dir/{out_dir}/alignment_dict.json", MyAligner.alignment_dict)
    align_dict = utils.read_json(f"result_dir/{out_dir}/alignment_dict.json")

    # Let's merge each alignment table into one and inject the omissions
    list_of_merged_alignments = graph_merge.merge_alignment_table(align_dict)

    # TODO: re-run the alignment on the units that are absent in the base wit.  
    

    # On teste si on ne perd pas de noeuds textuels
    print("Testing results consistency")
    possible_witnesses = string.ascii_lowercase[:len(align_dict) + 1]
    utils.test_tables_consistency(list_of_merged_alignments, possible_witnesses)
    # TODO: une phase de test pour voir si l'alignement final est cohérent avec les alignements deux à deux
    
    
    # Let's save the final tables (indices and texts)
    MyAligner.save_final_result(merged_alignments=list_of_merged_alignments, file_titles=sys.argv[1:-3])
    

if __name__ == '__main__':
    run_alignments()
                
            
    