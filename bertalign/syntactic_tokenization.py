import copy
import random
import re
import json
import sys
import langdetect 


def syntactic_tokenization(path):
    with open(path, "r") as input_text:
        text = input_text.read().replace("\n", " ")

    codelang = langdetect.detect(text[:300])
    with open("bertalign/delimiters.json", "r") as input_json:
        dictionary = json.load(input_json)
    
    single_tokens_punctuation = [punct for punct in dictionary[codelang]['punctuation'] if len(punct) == 1]
    multiple_tokens_punctuation = [punct for punct in dictionary[codelang]['punctuation'] if len(punct) != 1]
    single_token_punct = "".join(single_tokens_punctuation)
    multiple_tokens_punct = "|".join(multiple_tokens_punctuation)
    punctuation_subregex = f"{multiple_tokens_punct}|[{single_token_punct}]"
    tokens_subregex = "(" + " | ".join(dictionary[codelang]['word_delimiters']) + "|" + punctuation_subregex + ")"
    delimiter = re.compile(tokens_subregex)
    search = re.search(delimiter, text)
    tokenized_text = []
    if search:
        matches = re.split(delimiter, text)
    tokenized_text.append(matches[0])
    tokenized_text.extend([matches[index] + matches[index+1] for index in range(1, len(matches[:-1]), 2)])
    text_copy = copy.deepcopy(tokenized_text)
    random.shuffle(text_copy)
    tokenized_text = [token.strip() for token in tokenized_text]
    # for index, match in enumerate(matches):
    #     search = re.search(delimiter, match)
    #     if search:
    #         print(search)
    #     else:
    #         print("NO")

    with open("/home/mgl/Documents/test/split.txt", "w") as output_file:
        output_file.write("\n".join(tokenized_text))
    return tokenized_text
    
            
            
if __name__ == '__main__':
    syntactic_tokenization(sys.argv[1])