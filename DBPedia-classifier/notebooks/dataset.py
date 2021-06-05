"""02. PyTorch Dataset class - Data Preprocessing, Iterator and Item fetcher"""

import torch
import pickle
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from typing import List, Tuple


class DBPDataset(Dataset):

    def _load_dictionary(self):
        
        """create an obj to dictionary class and construct the dictionary"""
        file = open("C:/Users/rahin/projects/nn-pytorch-examples/DBPedia-classifier/data/interim/complete_word_dict.pkl", "rb")
        dict_content = pickle.load(file)
        file.close()

        return dict_content

    def _dataset_parser(self, inputdataset) -> Tuple[List, List]:

        """01. Token -> ID conversion pipelines"""

        def check_token(x):
            for token in x:
                if token not in self.vocabulary.keys():
                    self.vocabulary[token] = len(self.vocabulary)+1

            return self.vocabulary

        def sample_pipeline(x):
            
            vocab = check_token(x)
            return [vocab[token] for token in x]

            #return [self.vocabulary[token] for token in x]

        def label_pipeline(x):
            return int(x)-1

        tokenizer = get_tokenizer('basic_english')
        samples, labels = [], []

        for (label,line) in inputdataset:

            tokens = tokenizer(line)

            if len(tokens) > 249:
                continue

            for pad in range(0, 250-len(tokens)):
                tokens.append('PAD')

            word_embedding = sample_pipeline(tokens)

            # for pad in range(0, 250-len(word_embedding)):
            #     word_embedding.append(0)
            
            current_sample = torch.tensor(word_embedding, dtype=torch.int64)
            samples.append(current_sample)
            
            label_embedding = label_pipeline(label)
            current_label = torch.tensor(label_embedding, dtype=torch.int64)
            labels.append(current_label)

        return samples, labels

    
    def __init__(self, inputdataset: str):
        print("Building Dictionary...")
        self.vocabulary = self._load_dictionary()
        print("Done...")
        print("Building Dataset...")
        self.samples, self.labels = self._dataset_parser(inputdataset)
        print("Done...")

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):
        #print(f"Length of samples: {len(self.samples)} and Length of labels: {len(self.labels)}")

        return self.samples[idx], self.labels[idx]
