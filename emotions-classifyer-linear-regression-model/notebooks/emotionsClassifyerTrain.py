"""03. Pytorch Dataset(Train/Test/Validation) DataLoader and Model training"""
import pickle
import time
import torch
from torch.utils.data import DataLoader
from emotionsClassifyerDataset import myDataset
from emotionsClassifyerModel import EmotionsClassificationModel

# Step 1: Load Train, Test and Validation Datasets 
validation_dataset = DataLoader(dataset = myDataset("./nn-pytorch-examples/emotions-classifyer-linear-regression-model/data/raw/val.txt",
                                "./nn-pytorch-examples/emotions-classifyer-linear-regression-model/data/interim/complete_word_dict.pkl",
                                "./nn-pytorch-examples/emotions-classifyer-linear-regression-model/data/interim/complete_emotions_dict.pkl"),
                                batch_size=128,
                                shuffle=False
                                ##collate_fn=lambda x: x
                                )

test_dataset = DataLoader(dataset = myDataset("./nn-pytorch-examples/emotions-classifyer-linear-regression-model/data/raw/test.txt",
                                "./nn-pytorch-examples/emotions-classifyer-linear-regression-model/data/interim/complete_word_dict.pkl",
                                "./nn-pytorch-examples/emotions-classifyer-linear-regression-model/data/interim/complete_emotions_dict.pkl"),
                                batch_size=128,
                                shuffle=False)                                

train_dataset = DataLoader(dataset = myDataset("./nn-pytorch-examples/emotions-classifyer-linear-regression-model/data/raw/train.txt",
                                "./nn-pytorch-examples/emotions-classifyer-linear-regression-model/data/interim/complete_word_dict.pkl",
                                "./nn-pytorch-examples/emotions-classifyer-linear-regression-model/data/interim/complete_emotions_dict.pkl"),
                                batch_size=128,
                                shuffle=False)

file = open("./nn-pytorch-examples/emotions-classifyer-linear-regression-model/data/interim/complete_word_dict.pkl", "rb")
complete_word_dict = pickle.load(file)
file.close()
file = open("./nn-pytorch-examples/emotions-classifyer-linear-regression-model/data/interim/complete_emotions_dict.pkl", "rb")
complete_emotions_dict = pickle.load(file)
file.close()

emotions_dict_reverse = {}

for key,value in complete_emotions_dict.items():
   emotions_dict_reverse[value] = key


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############## Model Parameters :- ###################
vocabulary_size = len(complete_word_dict)+1
embedding_dimension = 128 # size of a single input sample 
number_of_classes = len(complete_emotions_dict) # size of single output sample - n possible emotions

model = EmotionsClassificationModel(vocabulary_size=vocabulary_size, embedding_dimension = embedding_dimension, number_of_classes= number_of_classes)

print(model)
######################################################


def train(dataset_dataloader):

   model.train()
   total_accuracy, total_count = 0, 0
   log_interval = 500
   start_time = time.time()

   for idx, (sample, label) in enumerate(dataset_dataloader):

      #print(len(sample))
      # print(len(label))

      curr_sample = sample
      curr_label = torch.squeeze(label)  # squeeze: to remove redundant dimension
      #print(f"Label dimension: {label.size()} and current label dimension: {curr_label.size()}")
      optimizer.zero_grad()     
      predicted_label = model(curr_sample)

      #print(predicted_label)
      #print(predicted_label.size())
      #print(curr_label.size())

      loss = criterion(predicted_label, curr_label)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
      optimizer.step()
      total_accuracy += (predicted_label.argmax(1) == curr_label).sum().item()
      total_count += curr_label.size(0)

      if idx % log_interval == 0 and idx > 0:
         elapsed = time.time() - start_time
         print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}' .format(epoch, idx, len(dataset_dataloader),
                                              total_accuracy/total_count))
         total_accuracy, total_count = 0, 0
         start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (sample, label) in enumerate(dataloader):

            curr_sample = sample
            curr_label = torch.squeeze(label)

            predited_label = model(curr_sample)
            loss = criterion(predited_label, curr_label)
            total_acc += (predited_label.argmax(1) == curr_label).sum().item()
            total_count += curr_label.size(0)
    return total_acc/total_count

###################################################################################


# Hyperparameters
EPOCHS = 10 # epoch
LR = 5  # learning rate
BATCH_SIZE = 64 # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None


for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataset)
    accu_val = evaluate(validation_dataset)
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)
   



    
