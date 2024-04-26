# -*- coding: utf-8 -*-
import re
import torch
import evaluate
import numpy as np

# function to convert text in input as tokens and labels (if label is identified in the file, gives 1, in other cases, 0)
def convertToSentencesAndLabels(text):

	print("Converting to sentences and labels")

	sentencesList = []
	splitList = []
	
	for l in text:
		i = re.split('\n', l)[0]
		j = re.split('\$', i)

		sentenceAsList = re.split('\s', j[0])
		split= j[1]
		
		if ',' in split:
			splitOk = re.split(',', split)
		else:
			splitOk = [split]

		# case where token has a position, when there are several identical tokens in the sentence (and for ex. we want to get the 2nd one)
		for i in range(len(splitOk)):
			if '-' in splitOk[i]:
				position = re.split('-', splitOk[i])[1]
				splitOkk = re.split('-', splitOk[i])[0]
		localList = []
		liste = []

		for e in enumerate(sentenceAsList) :
			subList = []
			token = e[1]
			# test is on splitOKK (case where token has a position)
			if '-' in split and token == splitOkk:			 
				subList.append(token)
				liste.append(subList)
				longueur = len(liste)
				# if the lenght of the list = position, it is the good token, so we get 1
				if longueur == int(position):
					localList.append(1)
				else:
					localList.append(0)
						
						
			elif token in splitOk :
				localList.append(1)
			else:
				localList.append(0)
		
		sentence = j[0]
		sentencesList.append(sentence)
		splitList.append(localList)

	return sentencesList, splitList


# function to get the index of the tokens after BERT tokenization
def get_index_correspondence(sent, tokenizer):
	correspondence = [(0,0)]
	for word in sent:
		(raw_end, expand_end) = correspondence[-1]
		tokenized_word = tokenizer.tokenize(word)
		correspondence.append((raw_end+1, expand_end+len(tokenized_word)))
	return correspondence


# function to align labels between the tokens in input and the tokenized tokens
def align_labels(corresp, orig_labels):
	new_labels = [0 for r in range(corresp[-1][1])]
	for index, label in enumerate(orig_labels):
		# label which is interesting : 1
		if label == 1:
			### verbose ?
			#print(f"index is {index}")
			#print(f"label is {label}")
			#print(f"Corresp first subword is {corresp[index][1] + 1}")
			#print(f"Corresp first subword actual index is {corresp[index][1]}")
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
		self.tokenizer = tokenizer

	def __len__(self):
		return len(self.texts)

	def __getitem__(self, idx):
		# get the max length of the training set in order to have the good feature to put in tokenizer
		num_max_length = get_token_max_length(self.texts, self.tokenizer)
		print("Getting new item")
		# current text (one line, ie 12 tokens [before automatic BERT tokenization])
		text = self.texts[idx]
		# current labels for the line
		labels = self.labels[idx]
		print(text)
		print(labels)
		# tokenize the text with padding to get tensors with equal size (inserts 2, as for special tokens)
		# num_max_length is got supra
		toks = self.tokenizer(text, padding="max_length", max_length=num_max_length, truncation=True, return_tensors="pt")
		# get the text
		tokens = text.split()
		# get the index correspondences between text and tok text
		corresp = get_index_correspondence(tokens, self.tokenizer)
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
		print(f"New label {new_labels}")
		# return the results
		return {
			'input_ids': toks['input_ids'].squeeze(),
			'attention_mask': toks['attention_mask'].squeeze(),
			'labels': label
		}


def compute_metrics(eval_pred):
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
	print(predictions)
	print(labels)

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

	return {"accurracy": acc, "recall": recall_l, "precision": precision_l, "f1": f1_l}