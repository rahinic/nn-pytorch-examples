"""01. Word(Vocabulary) Dictionary"""

from torchtext.datasets import DBpedia
from torchtext.data.utils import get_tokenizer

class myDictionary:
    
    def get_dataset(self, dataset):

        """Get the samples from each dataset"""
        tokenizer = get_tokenizer('basic_english')
        words_in_dataset = []

        for idx, (label, line) in enumerate(dataset):
            tokens = tokenizer(line)
            words_in_dataset.append(tokens)

        return words_in_dataset   

    def build_vocab(self):

        """iterate through the samples, tokenize, construct a dictionary from unique list of words"""

        dataset1 = DBpedia(split="train")
        dataset2 = DBpedia(split="test")

        vocab_part1 = list(self.get_dataset(dataset1))
        vocab_part2 = list(self.get_dataset(dataset2))

        total_vocab = vocab_part1 + vocab_part2
        total_vocab_flat1 = [item for sublist in total_vocab for item in sublist]
        total_vocab_flat1.append('PAD')
        total_vocab_flat = list(set(total_vocab_flat1))

        dict = {}

        for idx, word in enumerate(total_vocab_flat):
            dict[word] = idx

        return dict

    def __init__(self):
        
        self.vocabulary = self.build_vocab()
