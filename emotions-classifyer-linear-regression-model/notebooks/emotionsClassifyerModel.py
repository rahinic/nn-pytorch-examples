"""04. Pytorch Emotions Classification Neural Network Model"""

from torch import nn
import torch

class EmotionsClassificationModel(nn.Module):

    """Simple Linear Regression Model"""

    def __init__(self, vocabulary_size, embedding_dimension, number_of_classes):
        super(EmotionsClassificationModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_dimension) #input layer
        self.fc1 = nn.Linear(embedding_dimension, 50) #hidden layer
        self.fc2 = nn.Linear(50, number_of_classes)         #output layer
        #self.init_weights()

    # def init_weights(self):
    #     initrange = 0.5
    #     self.embedding.weight.data.uniform_(-initrange, initrange)
    #     self.fc.weight.data.uniform_(-initrange, initrange)
    #     self.fc.bias.data.zero_() # do not accumulate weight to next sample.

    def forward(self, text):
        embedded = self.embedding(text)

        #print(embedded.size())
        sentence_embeddings = torch.mean(embedded, dim=1) #reduction operation
        #print(sentence_embeddings.size())

        return self.fc2(self.fc1((sentence_embeddings)))

        #return self.fc(embedded)[:,-1,:]