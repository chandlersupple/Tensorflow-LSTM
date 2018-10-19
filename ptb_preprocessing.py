import numpy as np

def preprocess():
    data = open('ptb.char.train.txt', 'r')
    ptb = data.readlines()
    
    vocabulary = [' ']
    data = []
    for line in range (len(ptb)):
        for char in range (len(ptb[line])):
            char_q = (ptb[line][char]).lower()
            if char_q != ' ':
                if char_q == '_':
                    data.append(' ')
                else:
                    data.append(char_q)
                    if char_q not in vocabulary:
                        vocabulary.append(char_q)

    rosetta = {}
    for term in range(len(vocabulary)):
        rosetta[vocabulary[term]] = term
        
    ptb_data = []
    for term in range(len(data)):
        ptb_data.append(rosetta.get(data[term]))
        
    return ptb_data, vocabulary, rosetta
