import copy
import os
import random
import re
import json
import sys
import langid 
import aquilign.align.utils as utils

def regex_tokenization(input_file:str, 
                           corpus_limit=None, 
                           use_punctuation=False, 
                           standalone=True, 
                           text=None,
                           lang=None):
    if standalone:
        with open(input_file, "r") as input_text:
            text = input_text.read().replace("\n", " ")
    else:
        pass
    # text = utils.normalize_text(text)
    with open("/content/multilingual-medieval-aligner-workshop/aquilign/preproc/delimiters.json", "r") as input_json:
        dictionary = json.load(input_json)
        
    codelang = lang
        
    try:
        dictionary[codelang]
    except KeyError:
        print(f"Language code {codelang} not present in the delimiters file. Please update the file.")
        exit(0)
    
    single_tokens_punctuation = [punct for punct in dictionary[codelang]['punctuation'] if len(punct) == 1]
    multiple_tokens_punctuation = [punct for punct in dictionary[codelang]['punctuation'] if len(punct) != 1]
    single_token_punct = "".join(single_tokens_punctuation)
    multiple_tokens_punct = "|".join(multiple_tokens_punctuation)
    punctuation_subregex = f"{multiple_tokens_punct}|[{single_token_punct}]"
    if use_punctuation:
        tokens_subregex = "( " + " | ".join(dictionary[codelang]['word_delimiters']) + " |" + punctuation_subregex + " )"
    else:
        tokens_subregex = "( " + " | ".join(dictionary[codelang]['word_delimiters']) + " )"
    delimiter = re.compile(tokens_subregex)
    search = re.search(delimiter, text)
    tokenized_text = []
    if search:
        matches = re.split(delimiter, text)
        tokenized_text.append(matches[0])
        tokenized_text.extend([matches[index] + matches[index+1] for index in range(1, len(matches[:-1]), 2)])
        tokenized_text = [token.strip() for token in tokenized_text]
    else:
        tokenized_text = [text]
    # Let's limit the length for test purposes
    if corpus_limit:
        tokenized_text = tokenized_text[:round(len(tokenized_text)*corpus_limit)]
    return tokenized_text

if __name__ == '__main__':
    input_file = sys.argv[1]
    tokens = regex_tokenization(sys.argv[1])
    with open(input_file.replace(".txt", ".tokenized.txt"), "w") as output_file:
        output_file.write("\n".join(tokens))
