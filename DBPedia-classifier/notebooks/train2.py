import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import DBPDataset
from torchtext.datasets import DBpedia
import pickle
import time
from model2 import RNNDBPediaClassifier
import torch.optim as optim

file = open("C:/Users/rahin/projects/nn-pytorch-examples/DBPedia-classifier/data/interim/complete_word_dict.pkl", "rb")
vocab = pickle.load(file)
file.close()

print(f"Length of vocabulary is: {len(vocab)}")
##########################################################################################
########################### Train & Test Datasets ########################################
train = DBpedia(split='train')
test =  DBpedia(split='test')

print("01. Loading the Test datasets:")
test_dataset = DataLoader(dataset=DBPDataset(inputdataset=test),
                          batch_size=256,
                          shuffle=True)
print("Done! Test Dataset loaded successfully!")
print("02. Loading the Train datasets:")
train_dataset = DataLoader(dataset=DBPDataset(inputdataset=train),
                          batch_size=256,
                          shuffle=True)
print("Done! Training Dataset loaded successfully!")
##########################################################################################
################################# 01.Model Parameters ####################################
VOCAB_SIZE = len(vocab)+1
EMBED_DIM = 100
HIDDEN_DIM = 32
NUM_LAYERS = 2
NUM_OF_CLASSES = 14
EPOCHS = 10
LEARNING_RATE = 5
BATCH_SIZE = 256
################################### 02. NN Model  #######################################
print("03. builing the model...")
model = RNNDBPediaClassifier(embedding_dimension= EMBED_DIM,
                            vocabulary_size=VOCAB_SIZE,
                            hidden_dimension=HIDDEN_DIM,
                            num_of_layers=NUM_LAYERS,
                            dropout=0.2,
                            output_dimension=NUM_OF_CLASSES)
print("----------------------------------------------------------------")
print("Done! here is our model:")
print(model)
print("----------------------------------------------------------------")
##########################################################################################
############################# 03. Optimizer and Loss  #################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()

#define metric
def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc
    
#push to cuda if available
model = model.to(device)
criterion = criterion.to(device)

##########################################################################################
############################## 04. NN Model Train Definition #############################

def train(model, dataset, optimizer, criterion):
    #log_interval = 500
    start_time = time.time()
    print(start_time)

    epoch_loss = 0
    epoch_accuracy = 0

    model.train()

    for idx, (sample, label) in enumerate(dataset):
       
       current_samples = sample
       current_labels = label

       optimizer.zero_grad()

       predicted_labels = model(current_samples)
       
       loss = criterion(predicted_labels, current_labels)
       accuracy = binary_accuracy(predicted_labels, current_labels)

       loss.backward()
       optimizer.step()

       epoch_loss += loss.item()
       epoch_accuracy += accuracy.item()

    return epoch_loss/len(dataset), epoch_accuracy/len(dataset)

##########################################################################################
################################ 05. NN Model Eval Definition ############################
def evaluate(model, dataset, criterion):
    
    start_time = time.time()
    print(start_time)

    epoch_loss = 0
    epoch_accuracy = 0
    model.eval()

    with torch.no_grad():

        for idx, (sample, label) in enumerate(dataset):
            current_samples = sample
            current_labels = label

            predicted_labels = model(current_samples)

            loss = criterion(predicted_labels, current_labels)
            accuracy = binary_accuracy(predicted_labels, current_labels)

            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()

    return epoch_loss/len(dataset), epoch_accuracy/len(dataset)

############################################################################################
################################## 06. NN Model Eval #####################################
N_EPOCHS = 5
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
     
    #train the model
    train_loss, train_acc = train(model, train_dataset, optimizer, criterion)
    
    #evaluate the model
    valid_loss, valid_acc = evaluate(model, test_dataset, criterion)
    
    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
