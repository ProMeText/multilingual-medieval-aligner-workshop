import tok_trainer_functions as functions
import sys
from transformers import BertTokenizer, AutoModelForTokenClassification
import re
import torch


def convertToSentencesAndLabels(text, tokenizer):
    print("Converting to sentences and labels")
    sentencesList = []
    splitList = []

    for l in text:
        print(l)
        i = re.split('\n', l)[0]
        j = re.split('\$', i)

        sentenceAsList = re.findall(r"[\.,;—:?!’'«»“/-]|\w+", j[0])
        split = j[1]

        if '£' in split:
            splitOk = re.split('£', split)
        else:
            splitOk = [split]

        # case where token has a position, when there are several identical tokens in the sentence (and for ex. we want to get the 2nd one)
        positionList = []
        tokenList = []
        for i in range(len(splitOk)):
            if re.search(r'-\d', splitOk[i]):
                position = re.split('-', splitOk[i])[1]
                # print(position)
                positionList.append(int(position))
                splitOkk = re.split('-', splitOk[i])[0]
                tokenList.append(splitOkk)
            else:
                pass

        localList = []

        # set tokenList to get all the concerned values
        tL = list(set(tokenList))
        # print(f'tl {type(tL)}')

        # prepare an emptyList with as empty sublists as concerned words
        emptyList = []
        for item in tL:
            localEmptyList = []
            emptyList.append(localEmptyList)

        for e in enumerate(sentenceAsList):

            # get just the token
            token = e[1]

            # if it is a word that separate and that correspond to same other tokens in the sentence
            if '-' in split and token in tokenList:

                # we get the position of the token in the set list
                postL = tL.index(token)

                # if it is the correct token
                if token == tL[postL]:
                    # we fill the empty list with the current token (empty list with a position that correspond to the position in the tL list
                    emptyList[postL].append(token)
                else:
                    pass

                # we get correspondant idx for tokenList ans positionList
                goodidx = [i for i, e in enumerate(tokenList) if e == token]

                # empty list in which we'll put the position we want to get for the correct token
                goodpos = []

                # we activate the position to get the correct element in positionList, based on idx
                for i in goodidx:
                    goodelem = positionList[i]
                    goodpos.append(int(goodelem))

                # if the actual len of the emptyList is in the list of the positions that interest us: we add one to the list
                if len(emptyList[postL]) in goodpos:
                    localList.append(1)
                else:
                    localList.append(0)


            elif token in splitOk:
                localList.append(1)
            else:
                localList.append(0)

        sentence = j[0]
        sentencesList.append(sentence)
        splitList.append(localList)

    num_max_length = get_token_max_length(sentencesList, tokenizer)
    out_toks_and_labels = []
    for text, labels in tqdm.tqdm(zip(sentencesList, splitList)):
        toks = tokenizer(text, padding="max_length", max_length=num_max_length, truncation=True,
                         return_tensors="pt")

        # get the text with the similar splits as for the creation of the data
        tokens = re.findall(r"[\.,;—:?!’'«»“/-]|\w+", text)
        # get the index correspondences between text and tok text
        corresp = get_index_correspondence(tokens, tokenizer)
        # aligning the label
        new_labels = align_labels(corresp, labels)
        # get the length of the tensor
        sq = (toks['input_ids'].squeeze())
        ### insert 2 for in the new_labels in order to get tensors with the same size !
        if len(sq) == len(new_labels):
            pass
        else:
            diff = len(sq) - len(new_labels)
            for elem in range(diff):
                new_labels.append(2)
        assert len(sq) == len(new_labels), "Mismatch"
        # tensorize the new labels
        label = torch.tensor(new_labels)
        out_toks_and_labels.append({'input_ids': toks['input_ids'].squeeze(),
                                    'attention_mask': toks['attention_mask'].squeeze(),
                                    'labels': label})
    return out_toks_and_labels


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
    print(as_list)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name, max_length=10)
    new_model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=3)
    # get the path of the default tokenizer
    

    toks_and_labels = functions.convertToSentencesAndLabels(as_list, tokenizer)
    for txt_example, gt in zip(as_list, toks_and_labels):
        # BERT-tok
        enco_nt_tok = tokenizer.encode(txt_example, truncation=True, padding=True, return_tensors="pt")
        # get the predictions from the model
        predictions = new_model(enco_nt_tok)
        preds = predictions[0]
        # apply the functions
        bert_labels = get_labels_from_preds(preds)
        gt_label_as_list = gt['label'].tolist()
        print(f"Text: {txt_example}")
        print(f"Predicted: {bert_labels}")
        print(f"Ground Truth: {gt_label_as_list}")
        print("---")
       
        
        


if __name__ == '__main__':
    file_to_test = sys.argv[1]
    model_path = sys.argv[2]
    tokenizer_name = sys.argv[3]
    num = int(sys.argv[4])
    test(file_to_test, model_path, tokenizer_name, num)