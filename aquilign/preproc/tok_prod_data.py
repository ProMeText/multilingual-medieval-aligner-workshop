# -*- coding: utf-8 -*-

import random
import os
from os import listdir
from os.path import join, isfile
import sys

## script which enables production of data which require to be completed
## produces two txt files : one for training, the other for evaluation
## the number of lines for both tasks need to be chosen, as the number of tokens in each line

## usage : python tok_prod_data.py folder_name num output_train_lines.txt num_train_lines output_eval_lines.txt num_eval_lines
## where : 
# folder_name is the name of the folder which contains the .txt files from we want to extract lines
# num is the number of tokens per line
# output_train_lines.txt is the name of the file which will contain train lines
# num_train_lines is the number of lines in this file
# output_eval_lines.txt is the name of the fil which will contain evaluation lines
# num_eval_lines is the number of lines in this file


# function of tokenization (get the chosen number of tokens per line)
def tokenize(text,num):
    words = text.split(" ")
    num = int(num)
    return [' '.join(words[i:i+num]) for i in range(0, len(words), num)]


if __name__ == '__main__':

	# folder
	texts_folder = sys.argv[1]
	directory = os.fsencode(texts_folder)

	listOfFiles = []

	# read all the files in the chosen folder
	for filename in listdir(directory):  
		full_path = join(directory, filename) 
		if isfile(full_path):  
			with open(full_path) as f:  
				textL = f.read().splitlines()
				localText = " ".join(str(element) for element in textL)
				listOfFiles.append(localText)

	# get all the text in one in order to execute the random choice
	text = " ".join(str(element) for element in listOfFiles)
	# the number of tokens per line
	numTok = sys.argv[2]
	# apply the tokenization
	text_as_list = tokenize(text,numTok)

	## train file
	output_train_file = sys.argv[3]
	num_sentences = int(sys.argv[4])
	random.seed(234)
	choice  = random.choices(text_as_list, k=num_sentences)
	docChoice = '\n'.join(choice)
	with open(output_train_file, 'w') as f:
		f.write(docChoice)

	## eval file
	output_eval_file = sys.argv[5]
	num_sentences_eval = int(sys.argv[6])
	random.seed(324)
	choiceEval  = random.choices(text_as_list, k=num_sentences_eval)
	docChoiceEval = '\n'.join(choiceEval)
	with open(output_eval_file, 'w') as f:
		f.write(docChoiceEval)