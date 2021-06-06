import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import DBPDataset
from torchtext.datasets import DBpedia
import pickle
import time
from model2 import RNNDBPediaClassifier
import torch.optim as optim
from torchtext.data.utils import get_tokenizer

file = open("nn-pytorch-examples/DBPedia-classifier/data/interim/complete_word_dict.pkl", "rb")
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
# optimizer = optim.Adam(model.parameters())
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

#define metric
def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    
    # correct = (rounded_preds == y).float() 
    _,pred_label = torch.max(rounded_preds, dim = 1)
    correct = (pred_label == y).float()
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
       #print(current_samples)
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
################################## 06. NN Model training #####################################
N_EPOCHS = 5
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    print(epoch)
     
    #train the model
    train_loss, train_acc = train(model, train_dataset, optimizer, criterion)
    
    #evaluate the model
    valid_loss, valid_acc = evaluate(model, test_dataset, criterion)
    
    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    
    print("-------------------------------------------------------------------")
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    print("-------------------------------------------------------------------")

############################################################################################
################################## 07. Model Predictions #####################################

DBpedia_label = {0: 'Company',
                1: 'EducationalInstitution',
                2: 'Artist',
                3: 'Athlete',
                4: 'OfficeHolder',
                5: 'MeanOfTransportation',
                6: 'Building',
                7: 'NaturalPlace',
                8: 'Village',
                9: 'Animal',
                10: 'Plant',
                11: 'Album',
                12: 'Film',
                13: 'WrittenWork'}


def predict(text, model, vocab):

    # line to tokens
    tokenizer = get_tokenizer("basic_english")
    tokens_in_line = tokenizer(text)

    # padding sequence
    for pad in range(0, 250-len(tokens_in_line)):
                tokens_in_line.append('PAD') 

    # padded tokens to idx look-up from vocabulary
    tokens_to_idx = [vocab[tok] for tok in tokens_in_line]

    # print(f"{tokens_in_line} \n {tokens_to_idx}")

    # token idx to tensor conversion
    idx_to_torch = torch.tensor(tokens_to_idx, dtype=torch.int64)
    idx_to_torch = idx_to_torch.unsqueeze(1).T


    with torch.no_grad():
        output = model(idx_to_torch)
        print(output)
        print(output.argmax(1))
        return output.argmax(1).item() 

model = model.to("cpu")

example = "sport news anyone?"

print("This is a %s news" %DBpedia_label[predict(example, model, vocab)])
