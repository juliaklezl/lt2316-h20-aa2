
import os
import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:


    def __init__(self, dump_folder="/tmp/aa2_models/"):
        self.dump_folder = dump_folder
        os.makedirs(dump_folder, exist_ok=True)


    def save_model(self, epoch, model, optimizer, loss, scores, hyperparamaters, model_name):
        # epoch = epoch
        # model =  a train pytroch model
        # optimizer = a pytorch Optimizer
        # loss = loss (detach it from GPU)
        # scores = dict where keys are names of metrics and values the value for the metric
        # hyperparamaters = dict of hyperparamaters
        # model_name = name of the model you have trained, make this name unique for each hyperparamater.  I suggest you name them:
        # model_1, model_2 etc 
        #  
        #
        # More info about saving and loading here:
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training

        save_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'hyperparamaters': hyperparamaters,
                        'loss': loss,
                        'scores': scores,
                        'model_name': model_name
                        }

        torch.save(save_dict, os.path.join(self.dump_folder, model_name + ".pt"))


    def load_model(self, model, model_name):  # had only self as argument - can we add more? - what does this function even do??
        # Finish this function so that it loads a model and return the appropriate variables
        self.model = model   # not sure about any of this, really.. copied from pytorch tutorial...
        self.model_name = model_name
        
        checkpoint = torch.load(os.path.join(self.dump_folder, model_name + ".pt"))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        pass


    def train(self, train_X, train_y, val_X, val_y, model_class, hyperparameters):
        # Finish this function so that it set up model then trains and saves it.
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y
        self.model_class = model_class
        self.hyperparameters = hyperparameters
        inputsize = list(train_X.shape)[2]  # = 7, number of features per word
        samplesize = list(train_X.shape)[1]
        outputsize = 5 # number of ner labels
        hiddensize = inputsize//2  # might change this, seems like a reasonable number..
        learning_rate = hyperparameters["learning_rate"]
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model_class(inputsize, hiddensize, outputsize, hyperparameters["number_layers"]).to(device) ## add other hyperparameters?
        print(device)
        batches = self.batching(train_X, train_y, batch_size = hyperparameters["batch_size"])
        batches_list = list(batches)
        #criterion = nn.CrossEntropyLoss()
        criterion = nn.MSELoss()
        #o = hyperparameters["optimizer"]
        #optimizer = optim.Adam(self.model.parameters())  # what are parameters?     
   
        self.model.train()
    
        if hyperparameters['optimizer'] == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate) 
        
        for e in range(hyperparameters["epochs"]):
            #print(len(batches_list))
            #print(batches_list)
            tot_loss = 0
            for batch, labels in batches_list:
            #    print(len(list(batch)))
            #    print(batch)
                for i in range(len(batch)):
            #        print("yes")
            #        print(sentence)
                    #print("batch")
                    #print(batch[i].shape)
                    #print(labels[i].shape)
            #        print("device after batch", sentence.get_device())
            #        print("tensor", sentence.type())
                    #tens = batch[i].float().view(samplesize, 1, inputsize)  
                    tens = batch[i].float().view(1, samplesize, inputsize)
                    #print(tens.shape)
                    optimizer.zero_grad()
                    #print(tens.shape)
                    pred = self.model(tens, device)
                    print(pred.shape)
                    print(pred)
                    print(torch.argmax(pred, dim=2).shape)
                    print(torch.argmax(pred, dim=2))
                    print(torch.argmax(pred, dim=2).flatten().shape)
                    print(torch.argmax(pred, dim=2).flatten())
                    print(labels[i])
                    #print(labels[i].shape)
                    #print(labels[i])
                    #preds = pred.permute(0, 2, 1)
                    loss = criterion(torch.argmax(pred, dim=2).flatten().float(), labels[i].float())
                    tot_loss += loss
                    loss.backward()
                    optimizer.step()
                print("Total loss in epoch {} is {}.".format(e, tot_loss))
        #self.save_model(e, self.model, optimizer, loss, scores, hyperparameters, model_name)
        
        
        
      
        pass
    
    def batching(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # this is ugly, make device overall
        permutation = torch.randperm(X.size()[0], device=device)
        permX = X[permutation]
        permy = y[permutation]
        splitX = torch.split(permX, batch_size)
        splity = torch.split(permy, batch_size)
        
        #print("splitx", splitX)
        #print("splitx shape", splitX.shape)
        #print("splity", splity)
        #print("splity shape", splity.shape)
        return zip(splitX, splity)
    
    
    
    def test(self, test_X, test_y, model_class, best_model_path):
        # Finish this function so that it loads a model, test is and print results.
        pass


    
    
#class Batcher:
#    def __init__(self, X, y, , batch_size=50, max_iter=None):
#        self.X = X
#        self.y = y
#        self.device = device
#        self.batch_size=batch_size
#        self.max_iter = max_iter
#        self.curr_iter = 0
#        
#    def __iter__(self):
#        return self
#    
#    def __next__(self):
#        if self.curr_iter == self.max_iter:
#            raise StopIteration
#        permutation = torch.randperm(self.X.size()[0], device=self.device)
#        permX = self.X[permutation]
#        permy = self.y[permutation]
#        splitX = torch.split(permX, self.batch_size)
#        splity = torch.split(permy, self.batch_size)
#        
#        self.curr_iter += 1
#        return zip(splitX, splity)



val loop:
    for split in test_batches:
                for sentence, label in split:
                 #   with torch.no_grad():
                    pred = model(sentence.float(), self.device)
                #pred_l = pred.tolist()
                #labels = label.tolist()
                #print("predshape", pred.shape)
                    for i in range(pred.shape[0]):
                            pred_s = pred[i]
                            label_s = label[i]
                    #print(pred_s.shape)
                            for j in range(len(pred_s)):
                                pred_t = int(torch.argmax(pred_s[j]))
                                label_t = int(label_s[j])
                                y_pred.append(pred_t)
                                y_label.append(label_t)