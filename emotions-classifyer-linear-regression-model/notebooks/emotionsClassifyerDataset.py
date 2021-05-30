"""02. PyTorch Dataset class - Data Preprocessing, Iterator and Item fetcher"""

import pickle
from typing import List, Tuple
import torch
from torch.utils.data import Dataset

class myDataset(Dataset):

    def _lookup_tables(self, dictfile):

        file = open(dictfile, "rb")
        dict_content = pickle.load(file)
        file.close()

        return dict_content

    def _load_file(self, inputfilepath: str):

        return open(inputfilepath, mode='r').readlines()

    def _file_parser(self,loaded_file_content) -> Tuple[List, List]:

        # Step (a): Text Tokens to Idx pipeline:
        
        def samples_pipeline(x):
            return [self.complete_word_dict[token] for token in x]

        def label_pipeline(x):
            return [self.complete_emotions_dict[x]]

        # Step (b): File Pre-processing and padding sequence:

        samples, labels = [], []

        TOTAL_WORDS_IN_SENTENCE = 100

        for line in loaded_file_content:

            line = line.split(';')

            current_sample = line[0].split(' ')
            length_of_padding_needed = TOTAL_WORDS_IN_SENTENCE - len(current_sample)

            #Padding
            if length_of_padding_needed > 0:
                for times in range(0,length_of_padding_needed):
                    current_sample.append('PAD')

            current_sample = torch.tensor(samples_pipeline(current_sample), dtype=torch.int64)
            samples.append(current_sample)

            current_label = line[1].split('\n')[0]
            current_label = torch.tensor(label_pipeline(current_label), dtype=torch.int64)
            labels.append(current_label)

        if(len(samples) == len(labels)):
            print("Length matches! Hurray!")
        else:
            print("Oops, something went wrong, check the dimension of samples and labels")

        return samples, labels

###########################################################################################
    def __init__(self, inputfile: str, dictfile1: str, dictfile2: str):

        #Step 1: Load the two look-up dictionaries from pickle files

        self.complete_word_dict = self._lookup_tables(dictfile= dictfile1)
        self.complete_emotions_dict = self._lookup_tables(dictfile= dictfile2)

        # Step 2: Load file and parse contents

        loaded_file = self._load_file(inputfilepath= inputfile)
        self.samples, self.labels = self._file_parser(loaded_file_content= loaded_file)

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        return self.samples[idx], self.labels[idx]