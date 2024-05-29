import tok_trainer_functions as functions
import sys
from transformers import BertTokenizer, AutoModelForTokenClassification
import re

def tokenize(text,num):
    words = text.split(" ")
    return [' '.join(words[i:i+num]) for i in range(0, len(words), num)]

def get_labels_from_preds(preds):
    bert_labels = []
    for pred in preds[-1]:
        label = [idx for idx, value in enumerate(pred) if value == max(pred)][0]
        bert_labels.append(label)
    return bert_labels

# correspondences between our labels and labels from the BERT-tok
def get_correspondence(sent, tokenizer):
    out = {}
    tokenized_index = 0
    for index, word in enumerate(sent):
        # print(tokenizer.tokenize(word))
        tokenized_word = tokenizer.tokenize(word)
        print(tokenized_word)
        out[index] = tuple(item for item in range(tokenized_index, tokenized_index + len(tokenized_word)))
        tokenized_index += len(tokenized_word)
    human_split_to_bert = out
    bert_split_to_human_split = {value: key for key, value in human_split_to_bert.items()}
    return human_split_to_bert, bert_split_to_human_split

def test(file, model_path, tokenizer_name, num):
    with open(file, "r") as input_file:
        as_list = [item.replace("\n", "") for item in input_file.readlines()]
    
    all_examples, all_labels = [], []
    
    new_model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=3)
    # get the path of the default tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name, max_length=10)
    for example in as_list:
        print(example)
        toks_and_labels = functions.convertToSentencesAndLabels(example, tokenizer)
        print(toks_and_labels)
        # BERT-tok
        enco_nt_tok = tokenizer.encode(example, truncation=True, padding=True, return_tensors="pt")
        # get the predictions from the model
        predictions = new_model(enco_nt_tok)
        preds = predictions[0]
        # apply the functions
        bert_labels = get_labels_from_preds(preds)
        print(bert_labels)
        


if __name__ == '__main__':
    file_to_test = sys.argv[1]
    model_path = sys.argv[2]
    tokenizer_name = sys.argv[3]
    num = int(sys.argv[4])
    test(file_to_test, model_path, tokenizer_name, num)