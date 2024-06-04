import sys
import re
import operator 


def convert_text_to_labels(tokenized_text):
    all_labels = []
    for idx, syntagm in enumerate(tokenized_text):
        labels = []
        splitted = syntagm.split()
        if idx == 0:
            [labels.append(0) for token in splitted]
        else:
            labels.append(1)
            [labels.append(0) for token in splitted[1:]]
        all_labels.append(labels)
    return all_labels


def convert_to_new_format(file, delimiter="£", save_file=True, max_len=30):
    """
    This function takes a \n separated text file and converts each token to the adapted format (each example as a line with
    some delimiters)
    """
    with open(file, "r") as input_file:
        as_list = [line.replace("\n", "") for line in input_file.readlines()]
    
    as_sentences = {}
    n = 0
    for line in as_list:
        try:
            as_sentences[n].append(line)
        except KeyError:
            as_sentences[n] = [line]
        if "." in line:
            n += 1

    examples = {}
    all_examples = []
    for key, example in as_sentences.items():
        out_example = {0:""}
        n = 0
        for syntagm in example:
            if len(out_example[n].split()) + len(syntagm) > max_len:
                n += 1
            try:
                if out_example[n] != "":
                    out_example[n] += f" {delimiter}{syntagm}"
                else:
                    out_example[n] = syntagm
            except KeyError:
                out_example[n] = syntagm
            out_example[n] = out_example[n].strip()
            
        filtered_dict = {}
        for key, value in out_example.items():
            if len(value.split()) < 15:
                continue
            else:
                filtered_dict[key] = value 
        all_examples.extend(filtered_dict.values())
    if save_file:
        with open(file.replace(".txt", ".formatted_new.txt"), "w") as output_file:
            output_file.write("\n".join(all_examples))
    
    return all_examples
    
    

def format(file, keep_punct, examples_length, save_file=True, standalone=True, tokenized_text:list=None, keep_dots=False):
    """
    Cette fonction produit à partir d'un fichier tokénisé à l'aide de retour charriot \n
    un fichier de ground truth pour la tokénisation
    """
    
    if keep_dots is False:
        punctuation_regex = r'[\.·;,:!¿?¡]'
    else:
        punctuation_regex = r'[·;,:!¿?¡]'
    if standalone:
        with open(file, "r") as input_file:
            as_list = [line.replace("\n", "") for line in input_file.readlines()]
            if keep_punct is False:
                as_list = [re.sub(punctuation_regex, "", line) for line in as_list]
    else:
        as_list = tokenized_text
    
    as_sentences = {}
    n = 0
    # On a deux façons de comprendre le terme "exemple". Un exemple en entrée de fonction, 
    # c'est une phrase. Un exemple en sortie de fonction, c'est une suite de mots de longueur n.
    # On splitte les exemples et on le met dans un dictionnaire
    for line in as_list:
        try:
            as_sentences[n].append(line)
        except KeyError:
            as_sentences[n] = [line]
        if "." in line:
            n += 1
    
    # On va itérer sur chaque exemple splitté
    examples = {}
    n = 0
    for key, sentence in as_sentences.items():
        as_string = " ".join(sentence)
        splitted = as_string.split()
        clusters = {}
        all_delimiters = [len(sent.split()) for sent in sentence]
        delim_indices = []
        previous_value = 0
        out_list = []
        delim_indices = []
        
        # On crée la liste des position des délimiteurs exemple par exemple
        for index, element in enumerate(all_delimiters[:-1]):
            delim_indices.append(element + previous_value)
            previous_value = element + previous_value
        updated_index = 0 
        # On va créer un dictionnaire de forme {index: {"labels": [...], "text": [...]}}
        for index, token in enumerate(splitted):
            if updated_index != 0 and updated_index % examples_length == 0:
                n += 1
            try: 
                examples[n]["text"].append(token) 
            except KeyError:
                examples[n] = {"text": [token], "labels": []}
            
            # Gestion des délimiteurs
            if index in delim_indices:
                # On va chercher la position dans la phrase pour désambiguiser
                sublist = examples[n]["text"][:-1]
                position_multiple = operator.countOf(sublist, token) + 1
                token_with_count = f"{token}-{position_multiple}"
                examples[n]["labels"].append(token_with_count)
                
            # Changement d'exemple s'il y a changement de phrase
            if "." in token:
                n += 1
                updated_index = 0
            updated_index += 1
    
    
    out_list = []
    for example in examples.values():
        text = " ".join(example['text'])
        labels = ""
        if example["labels"] != [] and len(example["labels"]) != 1:
            labels = f"${example['labels'][0]}" + "£" + '£'.join(example['labels'][1:])
        elif example["labels"] != [] and len(example["labels"]) == 1:
            labels = f"${example['labels'][0]}"
        elif example["labels"] == []:
            continue
        out_list.append(text+labels)
        
    if save_file:
        with open(file.replace(".txt", ".formatted.txt"), "w") as output_file:
            output_file.write("\n".join(out_list))
    return out_list
            
if __name__ == '__main__':
    file_to_create = sys.argv[1]
    keep_punct = sys.argv[2]
    punct = keep_punct == "True"
    keep_dot = sys.argv[3]
    dot = keep_dot == "True"
    examples_length = 50
    format(file_to_create, punct, examples_length, keep_dot)
    convert_to_new_format(file_to_create, max_len=examples_length)