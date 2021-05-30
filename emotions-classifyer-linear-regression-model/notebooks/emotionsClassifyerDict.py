"""01. Word and Emotions(labels) look-up dictionaries"""

from typing import List, Tuple
import pickle

def dataset_and_labels(open_file) -> Tuple[List, List]:

    """Perform File pre-processing to split the contents into samples and corresponding labels"""

    dataset_from_file, labels_from_file = [], []

    #df = pd.DataFrame()

    for line in open_file:
        line = line.split(';')
        dataset_from_file.append(line[0])
        labels_from_file.append(line[1].split('\n')[0])

    return dataset_from_file, labels_from_file


def read_corpus(input_filename):

    """Open and read the contents of the given file"""

    print(f"\nLoading the file {input_filename}...")
    filename = open(input_filename, mode='r').readlines()
    print("\nFile Load Complete! Let's process it to identify labels")
    dataset, labels = dataset_and_labels(filename)
    print(f"Done!")

    return dataset, labels

list_of_files = ["./nn-pytorch-examples/emotions-classifyer-linear-regression-model/data/raw/train.txt",
                "./nn-pytorch-examples/emotions-classifyer-linear-regression-model/data/raw/test.txt",
                "./nn-pytorch-examples/emotions-classifyer-linear-regression-model/data/raw/val.txt"]

all_samples, all_labels = [], []

for file in list_of_files:
    curr_sample, curr_labels = read_corpus(file)
    all_samples.append(curr_sample)
    all_labels.append(curr_labels)

#list of list flattening
samples = [item for sublist in all_samples for item in sublist]
labels = [item for sublist in all_labels for item in sublist]


def word_label_lookup_dict(text,classification):

    """Word tokenization + list of classes, enumeration and dictionary construction"""

    class_dict = {}
    word_dict = {}
    word_list = []
    #unique_words = list(set(samples))
    unique_classes = list(set(labels))

    for idx, label in enumerate(unique_classes):
        class_dict[label] = idx

    for line in samples:
        word_list.append(line.split(' '))
    
    unique_word_list = [item for sublist in word_list for item in sublist]
    print(len(unique_word_list))
    unique_word_list = list(set(unique_word_list)) # unique list of words in all three files

    for idx, word in enumerate(unique_word_list):
        #print(word)
        word_dict[word] = idx

    print(dict(list(word_dict.items())[:20]))
    print('\n')
    print(dict(list(class_dict.items())[:20]))
    print('\n')

    #padding_sequence
    word_dict['PAD'] = len(word_dict)+1

    return word_dict, class_dict

word_dict, class_dict = word_label_lookup_dict(samples,labels)
        
# Look-up tables export:
dict_to_file = open("./nn-pytorch-examples/emotions-classifyer-linear-regression-model/data/interim/complete_word_dict.pkl", "wb")
pickle.dump(word_dict, dict_to_file)
dict_to_file.close()

dict_to_file = open("./nn-pytorch-examples/emotions-classifyer-linear-regression-model/data/interim/complete_emotions_dict.pkl", "wb")
pickle.dump(class_dict, dict_to_file)
dict_to_file.close()
