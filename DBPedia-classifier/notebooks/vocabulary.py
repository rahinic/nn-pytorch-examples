import pickle
from dictionary import myDictionary

#construct the dictionary
myDict = myDictionary()
vocabulary = myDict.vocabulary

#export the dictionary to pickle
dict_to_file = open("C:/Users/rahin/projects/nn-pytorch-examples/DBPedia-classifier/data/interim/complete_word_dict.pkl", "wb")
pickle.dump(vocabulary, dict_to_file)
dict_to_file.close()