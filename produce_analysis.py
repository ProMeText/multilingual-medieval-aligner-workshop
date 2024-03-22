import tqdm
import itertools
import bertalign.utils as utils
from bertalign import Bertalign

def create_list(path_to_csv):
    """
    This function computes the similarities of each alignment unit from a single csv file
    """
    with open(path_to_csv, "r") as input_csv:
        csv_file = input_csv.read()
    csv_list = csv_file.split("\n")
    csv_list = [row.split(",") for row in csv_list]
    alignment_dict = {index: alignement[1:] for index, alignement in enumerate(csv_list[1:])}
    return alignment_dict

def compute(text_a, text_b):
    aligner = Bertalign([text_a[1]], [text_b[1]], max_align=3)
    output = aligner.compute_distance()
    print(text_a)
    print(text_b)
    print(output)
    if output > .6:
        print("Texts seem similar")
    else:
        print("Texts seem disimilar")
    return output
        

def compute_similarity(alignments:list):
    combinaisons = list(set(itertools.combinations(alignments, 2)))
    current_dict = {}
    for combinaison in combinaisons:
        cosine_sim = compute(combinaison[0], combinaison[1])
        current_dict[f"{combinaison[0][0]}-{combinaison[1][0]}"] = cosine_sim.item()
    return current_dict
    
    
    
def main(path):
    alignment_dict = create_list(path)
    alignments_as_similarities = []
    for index, alignments in tqdm.tqdm(alignment_dict.items()):
        print("\nNew alignment unit")
        non_empty_entries = len([(index, element) for index, element in enumerate(alignments) if element != ""])
        if non_empty_entries != 1:
            alignments_as_similarities.append(compute_similarity([(index, element.replace("|", " ")) for index, element in enumerate(alignments) if element != ""]))
        else:
            alignments_as_similarities.append([])
    print(alignments_as_similarities)
    utils.write_json("result_dir/lancelot_1/similarities_as_list.json", alignments_as_similarities)
        
        
    
    
    
if __name__ == '__main__':
    main("result_dir/lancelot_1/final_result.csv")