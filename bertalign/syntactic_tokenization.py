import copy
import os
import random
import re
import json
import sys
import langdetect 
import bertalign.utils as utils

def syntactic_tokenization(path, corpus_limit=None):
    name = path.split("/")[-1].split(".")[0]
    with open(path, "r") as input_text:
        text = input_text.read().replace("\n", " ")

    text = utils.normalize_text(text)
    codelang = langdetect.detect(text[:300])
    with open("bertalign/delimiters.json", "r") as input_json:
        dictionary = json.load(input_json)
    
    single_tokens_punctuation = [punct for punct in dictionary[codelang]['punctuation'] if len(punct) == 1]
    multiple_tokens_punctuation = [punct for punct in dictionary[codelang]['punctuation'] if len(punct) != 1]
    single_token_punct = "".join(single_tokens_punctuation)
    multiple_tokens_punct = "|".join(multiple_tokens_punctuation)
    punctuation_subregex = f"{multiple_tokens_punct}|[{single_token_punct}]"
    tokens_subregex = "(" + " | ".join(dictionary[codelang]['word_delimiters']) + " |" + punctuation_subregex + ")"
    delimiter = re.compile(tokens_subregex)
    search = re.search(delimiter, text)
    tokenized_text = []
    if search:
        matches = re.split(delimiter, text)
    tokenized_text.append(matches[0])
    tokenized_text.extend([matches[index] + matches[index+1] for index in range(1, len(matches[:-1]), 2)])
    tokenized_text = [token.strip() for token in tokenized_text]

    # Let's limit the length for test purposes
    if corpus_limit:
        tokenized_text = tokenized_text[:corpus_limit]
    # for index, match in enumerate(matches):
    #     search = re.search(delimiter, match)
    #     if search:
    #         print(search)
    #     else:
    #         print("NO")
    
    try:
        os.mkdir("result_dir")
    except FileExistsError:
        pass
    with open(f"result_dir/split_{name}.txt", "w") as output_file:
        output_file.write("\n".join(tokenized_text))
    utils.write_json(f"result_dir/split_{name}.json", tokenized_text)
    return tokenized_text
    
            
            
if __name__ == '__main__':
    syntactic_tokenization(sys.argv[1])