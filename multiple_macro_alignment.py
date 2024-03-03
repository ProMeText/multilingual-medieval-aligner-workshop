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

def create_pairs(full_list:list, main_wit_index:int):
    pairs = []
    main_wit = full_list.pop(int(main_wit_index))
    for wit in full_list:
        pairs.append((main_wit, wit))
    print(pairs)
    return pairs

def parallel_align(first_alignment):
    pairs = create_pairs(sys.argv[:1], 0)
    alignment_dict = dict()
    for index, (main_wit, wit_to_compare) in enumerate(pairs):
        first_tokenized_text = syntactic_tokenization.syntactic_tokenization(main_wit)
        second_tokenized_text = syntactic_tokenization.syntactic_tokenization(wit_to_compare)
        aligner = Bertalign(first_tokenized_text, second_tokenized_text)
        aligner.align_sents(first_alignment_only=first_alignment)
        alignment_dict[index] = aligner.result
        write_json(f"result_dir/alignment_1_{str(index)}.json", aligner.result)
        
    utils.save_alignment_results(aligner.result, first_tokenized_text, second_tokenized_text, f"{main_wit}_{wit_to_compare}")
    


def merge_align():
    import bertalign.syntactic_tokenization as syntactic_tokenization
    first_text = syntactic_tokenization.syntactic_tokenization(sys.argv[1])
    second_text = syntactic_tokenization.syntactic_tokenization(sys.argv[2])
    troisieme_texte = syntactic_tokenization.syntactic_tokenization(sys.argv[3])
    aligner_1 = Bertalign(first_text, second_text)
    
    # On va tâcher de fusionner les résultats de la première phase d'alignement de Bertalign
    aligner_1.align_sents(first_alignment_only=True)
    aligner_2 = Bertalign(first_text, troisieme_texte)
    aligner_2.align_sents(first_alignment_only=True)
    print(aligner_1.result)
    print(aligner_2.result)
    write_json("/home/mgl/Documents/test/alignment_1.json", aligner_1.result)
    write_json("/home/mgl/Documents/test/alignment_2.json", aligner_2.result)
    merged = list(zip(aligner_1.result, aligner_2.result))
    with open("/home/mgl/Documents/test/merged.json", "w") as output_file:
        json.dump(merged, output_file, cls=NumpyEncoder)
    # aligner.print_sents()
    write_json("/home/mgl/Documents/test/texts.json", (first_text, second_text, troisieme_texte))

def blue_print(string):
    OKBLUE = '\033[94m'
    ENDC = '\033[0m'
    print(f"{OKBLUE}{string}{ENDC}")



def red_print(string):
    RED = '\033[31m'
    ENDC = '\033[0m'
    print(f"{RED}{string}{ENDC}")


def merge_lists(length):
    texts = read_json("/home/mgl/Documents/test/texts.json")
    
    with open("/home/mgl/Documents/test/texts.csv", "w") as output_text:
        for position in range(length):
            output_text.write('\t'.join([texts[index][position] for index in range(len(texts))]))
            output_text.write("\n")
    
    liste_1 = read_json("/home/mgl/Documents/test/alignment_1.json")
    liste_2 = read_json("/home/mgl/Documents/test/alignment_2.json")
    with open("/home/mgl/Documents/test/merged.json", "r") as output_file:
        file = json.load(output_file)
    output_list = []
    for item in range(length):
        output_list.append([item])
        for index, liste in enumerate([liste_1, liste_2]):
            if any(element[0] == item for element in liste):
                correct_index = [element[1] for element in liste if element[0] == item][0]
                output_list[item].append(correct_index)
            else:
                output_list[item].append(-1)
    print(output_list)
    
    as_array = np.array(output_list)
    
    
    # Le but ici est d'insérer les correspondances vides dans la liste finale
    for item in range(length):
        if item in list(as_array[...,1]):
            pass
        else:
            correct_index = [index + 2 for index, value in enumerate(list(as_array[...,1])[1:]) if value == item - 1][0]
            as_array = np.insert(as_array, correct_index, np.array([-1, item, -1]), axis=0)
    for item in range(41):
        if item in list(as_array[...,2]):
            pass
        else:
            correct_index = [index + 2 for index, value in enumerate(list(as_array[...,1])[1:]) if value == item - 1][0]
            as_array = np.insert(as_array, correct_index, np.array([-1, -1, item]), axis=0)
            
            
    array_to_list = list(as_array)
    output_list_as_text = []
    for out_index, item in enumerate(array_to_list):
        item = list(item)
        for index in range(len(texts)):
            if list(array_to_list[out_index])[index] == -1:
                try:
                    output_list_as_text[out_index].append("ø")
                except IndexError:
                    output_list_as_text.append(["ø"])
            else:
                correct_index = item[index]
                try:
                    output_list_as_text[out_index].append(texts[index][correct_index])
                except IndexError:
                    output_list_as_text.append([texts[index][correct_index]])
    
    print(output_list_as_text)
    print(len(output_list_as_text))
    
    with open("/home/mgl/Documents/test/merged.tsv", "w") as output_file:
        for line in list(as_array):
            output_file.write("\t".join([str(item) for item in line]))
            output_file.write("\n")
    with open("/home/mgl/Documents/test/merged_text.tsv", "w") as output_file:
        for line in output_list_as_text:
            output_file.write("\t".join([str(item) for item in line]))
            output_file.write("\n")
            
            
def find_in_list_of_tuples(value, list_of_tuples):
    for index, item in enumerate(list_of_tuples):
        target, aligned = item
        if value in target:
            return index
        
def merge(alignment_a, alignment_b):
    end_list = []
    current_pos_out = 0
    follow_out = True
    while follow_out:
        last_position = 0
        red_print(current_pos_out)
        red_print(f"New alignment unit: {alignment_a[current_pos_out]}")
        base_wit_out, compared_wit_a = alignment_a[current_pos_out]
        print(f"base wit: {base_wit_out}")
        current_pos_in = 0
        follow_in = True
        while follow_in:
            interm_list = []
            blue_print(f"Comparing with: {alignment_b[current_pos_in]}")
            base_wit_in, compared_wit_b = alignment_b[current_pos_in]
            # CAS 1: correspondance exacte
            # Si on a une correspondance exacte, c'est encore le plus simple.
            if base_wit_out == base_wit_in:
                print("Exact match. Merging the two alignment tables")
                print(alignment_b[current_pos_in])
                interm_list.extend([base_wit_out, compared_wit_a, compared_wit_b])
                end_list.append(interm_list)
                follow_in = False
            elif any(pos in base_wit_in for pos in base_wit_out):
                print(f"Tokens mergins tokens: {base_wit_in}")
                # Quand la fusion s'est faite sur la table d'alignement 2
                if len(base_wit_out) < len(base_wit_in):
                    print("Table B merged multiple tokens")
                    correct_positions = []
                    for pos in base_wit_in:
                        corresponding = find_in_list_of_tuples(pos, alignment_a)
                        print(corresponding)
                        correct_positions.append(corresponding)
                    print(correct_positions)
                    interm_2_list = []
                    for correct_position in correct_positions:
                        interm_2_list.extend(alignment_a[correct_position][1])
                    last_position = find_in_list_of_tuples(correct_positions[-1], alignment_a)
                    print(last_position)
                    print(interm_2_list)
                    interm_list.append([interm_2_list, base_wit_in, compared_wit_b])
                print(interm_list)
                end_list.extend(interm_list)
                follow_in = False
            else:
                print("Nothing found. Passing")
                pass
            if current_pos_in + 1 < len(alignment_b):
                current_pos_in += 1
            else:
                follow_in = False
        # if current_pos_out + 1 < len(alignment_a):
        if current_pos_out + 1 < 4:
            if last_position == 0:
                current_pos_out += 1 + last_position
            else:
                current_pos_out += last_position
        else:
            follow_out = False
        print(end_list)
        

class Aligner:
    def __init__(self, corpus_size:None, max_align=3):
        self.alignment_dict = dict()
        self.text_dict = dict()
        self.files_path = sys.argv[1:-1]
        self.main_file_index = sys.argv[-1]
        self.corpus_size = corpus_size
        self.max_align = max_align

    def parallel_align(self):
        pairs = create_pairs(self.files_path, self.main_file_index)
        for index, (main_wit, wit_to_compare) in enumerate(pairs):
            main_wit_name = main_wit.split("/")[-1].split(".")[0]
            wit_to_compare_name = wit_to_compare.split("/")[-1].split(".")[0]
            print(f"Aligning {main_wit} with {wit_to_compare}")
            first_tokenized_text = utils.clean_tokenized_content(syntactic_tokenization.syntactic_tokenization(main_wit, corpus_limit=self.corpus_size))
            print(len(first_tokenized_text))
            second_tokenized_text = utils.clean_tokenized_content(syntactic_tokenization.syntactic_tokenization(wit_to_compare, corpus_limit=self.corpus_size))
            try:
                os.mkdir("result_dir")
            except FileExistsError:
                pass
            utils.write_json(f"result_dir/split_{wit_to_compare_name}.json", second_tokenized_text)
            self.text_dict[0] = first_tokenized_text
            self.text_dict[index + 1] = second_tokenized_text
            aligner = Bertalign(first_tokenized_text, second_tokenized_text, max_align= self.max_align)
            aligner.align_sents()
            self.alignment_dict[index] = aligner.result
            utils.write_json(f"result_dir/alignment_{str(index)}.json", aligner.result)
            utils.save_alignment_results(aligner.result, first_tokenized_text, second_tokenized_text,
                                         f"{main_wit_name}_{wit_to_compare_name}")
        utils.write_json(f"result_dir/alignment_dict.json", self.alignment_dict)

def save_final_result(list_of_merged_alignments, MyAligner):
    with open("result_dir/final_result.tsv", "w") as output_text:
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
    MyAligner = Aligner(corpus_size=None, max_align=3)
    MyAligner.parallel_align()
    utils.write_json("result_dir/alignment_dict.json", MyAligner.alignment_dict)
    align_dict = utils.read_json("result_dir/alignment_dict.json")
    list_of_merged_alignments = graph_merge.merge_alignment_table(align_dict)
    # On teste si on ne perd pas de noeuds textuels
    utils.test_tables_consistency(list_of_merged_alignments, 'abcde')
    save_final_result(list_of_merged_alignments, MyAligner)
    
    
                
            
    