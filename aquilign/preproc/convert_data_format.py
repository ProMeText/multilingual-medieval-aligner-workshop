import sys
import re 
def main(file, delimiter, verbose=True):
    with open(file, "r") as input_file:
        examples = [item.replace("\n", "") for item in input_file.readlines()]

    sentencesList = []
    splitList = []
    formatted_examples = []
    for example in examples:
        j = re.split('\$', example)
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
                positionList.append(int(position))
                splitOkk = re.split('-', splitOk[i])[0]
                tokenList.append(splitOkk)
            else:
                pass

        localList = []
        tL = list(set(tokenList))
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
        assert len(sentenceAsList) == len(localList), "length differ, please check code"
        formatted_example = [f"{delimiter}{token}" if localList[idx] == 1 else token for idx, token in enumerate(sentenceAsList)]
        formatted_example = " ".join(formatted_example)
        formatted_examples.append(formatted_example)
        if verbose:
            print(example)
            print(sentenceAsList)
            print(formatted_example)
        
        
    with open(file.replace(".txt", ".reformatted.txt"), "w") as output_file:
        output_file.write("\n".join(formatted_examples))

    return sentencesList, splitList


if __name__ == '__main__':
    input_file = sys.argv[1]
    delimiter = "£"
    main(input_file, delimiter=delimiter)