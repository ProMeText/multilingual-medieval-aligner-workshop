# -*- coding: utf-8 -*-
import re
import torch
import evaluate
import numpy as np
import tqdm 
# function to convert text in input as tokens and labels (if label is identified in the file, gives 1, in other cases, 0)
def convertToSentencesAndLabels(text, tokenizer):

    print("Converting to sentences and labels")

    sentencesList = []
    splitList = []

    for l in text:
        print(l)
        i = re.split('\n', l)[0]
        j = re.split('\$', i)

        sentenceAsList = re.findall(r"[\.,;—:?!’'«»“/-]|\w+", j[0])
        split= j[1]

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
                print(position)
                positionList.append(int(position))
                splitOkk = re.split('-', splitOk[i])[0]
                tokenList.append(splitOkk)
            else:
                pass


        localList = []

        #set tokenList to get all the concerned values
        tL = list(set(tokenList))
        print(f'tl {type(tL)}')

        #prepare an emptyList with as empty sublists as concerned words
        emptyList = []
        for item in tL:
            localEmptyList = []
            emptyList.append(localEmptyList)

        for e in enumerate(sentenceAsList) :

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
    out_toks, out_labels = [], []
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
        out_toks.append(toks)
        out_labels.append(label)
    return out_toks, out_labels




# function to get the index of the tokens after BERT tokenization
def get_index_correspondence(sent, tokenizer):
    correspondence = [(0,0)]
    for word in sent:
        (raw_end, expand_end) = correspondence[-1]
        tokenized_word = tokenizer.tokenize(word)
        # print(tokenized_word)
        correspondence.append((raw_end+1, expand_end+len(tokenized_word)))
    return correspondence


# function to align labels between the tokens in input and the tokenized tokens
def align_labels(corresp, orig_labels):
    new_labels = [0 for r in range(corresp[-1][1])]
    for index, label in enumerate(orig_labels):
        # label which is interesting : 1
        if label == 1:
            ### verbose ?
            # print(f"index is {index}")
            # print(f"label is {label}")
            # print(f"Corresp first subword is {corresp[index][1] + 1}")
            # print(f"Corresp first subword actual index is {corresp[index][1]}")
            ###
            ### if the length of the new list = the current index
            if len(new_labels) == corresp[index][1]:
                new_labels[(corresp[index][1]) - 1] = 1
            else:
                new_labels[(corresp[index][1])] = 1
        else:
            pass
    # for special tokens (automatically added by BERT tokenizer), value of 2
    new_labels.insert(0, 2)
    new_labels.append(2)
    return new_labels


# function who gets the max length of tokenized text, used then in the class SentenceBoundaryDataset
def get_token_max_length(train_texts, tokenizer):
    lengths_list = []
    for text in train_texts:
        tok_text = tokenizer(text, return_tensors='pt')
        # get the length for every tok text
        tensor_length = (tok_text['input_ids'].squeeze())
        length = tensor_length.shape[0]
        lengths_list.append(length)
    # get the max value from the list
    max_length = max(lengths_list)
    return max_length


# dataset class which fits the requirements
class SentenceBoundaryDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # get the max length of the training set in order to have the good feature to put in tokenizer
        # current text (one line, ie 12 tokens [before automatic BERT tokenization])
        toks = self.texts[idx]
        # current labels for the line
        labels = self.labels[idx]
        return {
            'input_ids': toks['input_ids'].squeeze(),
            'attention_mask': toks['attention_mask'].squeeze(),
            'labels': labels
        }


def compute_metrics(eval_pred):
    print("Starting eval")
    # load the metrics we want to evaluate
    metric1 = evaluate.load("accuracy")
    metric2 = evaluate.load("recall")
    metric3 = evaluate.load("precision")
    metric4 = evaluate.load("f1")

    predictions, labels = eval_pred
    # get the label predictions
    predictions = np.argmax(predictions, axis=2)

    # get the right format
    predictions = np.array(predictions, dtype='int32').flatten()
    labels = np.array(labels, dtype='int32').flatten()

    # automatically, value of -100 are produce ; we haven't understood why but we change them to 0. If not, it will give poor results
    ###
    labels = [0 if x == -100 else x for x in labels]
    ###
    # print(predictions)
    # print(labels)

    acc = metric1.compute(predictions=predictions, references=labels)
    recall = metric2.compute(predictions=predictions, references=labels, average=None)
    recall_l = []
    [recall_l.extend(v) for k, v in recall.items()]
    precision = metric3.compute(predictions=predictions, references=labels, average=None)
    precision_l = []
    [precision_l.extend(v) for k, v in precision.items()]
    f1 = metric4.compute(predictions=predictions, references=labels, average=None)
    f1_l = []
    [f1_l.extend(v) for k, v in f1.items()]

    print("Eval finished")
    return {"accurracy": acc, "recall": recall_l, "precision": precision_l, "f1": f1_l}