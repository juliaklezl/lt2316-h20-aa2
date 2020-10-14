
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


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


    def load_model(self, model_path):  # had only self as argument - can we add more? - what does this function even do??
        # Finish this function so that it loads a model and return the appropriate variables
        self.model_path = model_path
        
        checkpoint = torch.load(model_path)
        return checkpoint
        
        #pass


    def train(self, train_X, train_y, val_X, val_y, model_class, hyperparameters):
        # Finish this function so that it set up model then trains and saves it.
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y
        self.model_class = model_class 
        
        lr = hyperparameters["learning_rate"]
        n_layers = hyperparameters["number_layers"]
        set_optimizer = hyperparameters["optimizer"]
        batch_size = hyperparameters["batch_size"]
        epochs = hyperparameters["epochs"]
        model_name = hyperparameters["model_name"]
        
        inputsize = list(train_X.shape)[2]  # = 7, number of features per word
        samplesize = list(train_X.shape)[1]
        outputsize = 5 # number of ner labels
        hiddensize = inputsize//2  # might change this, seems like a reasonable number..
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        train_batches = Batcher(train_X, train_y, self.device, batch_size=batch_size, max_iter=epochs)
        #test_batches = Batcher(val_X, val_y, self.device, batch_size=batch_size, max_iter=1)
        model = model_class(inputsize, hiddensize, outputsize, n_layers)
        model = model.to(self.device)
        
        
        if set_optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr) 
            
        criterion = nn.NLLLoss()
        #criterion = nn.MSELoss()
        model.train()
        
        epoch = 0
        for split in train_batches:
            model.train()
            tot_loss = 0
            for sentence, label in split:
                optimizer.zero_grad()
                pred = model(sentence.float(), self.device)
                #loss = criterion(torch.argmax(pred, dim=2).float(), label.float()).to(self.device)
                pred = pred.permute(0, 2, 1)        
                loss = criterion(pred.float(), label).to(self.device)
                tot_loss += loss
                loss.backward()
                optimizer.step()
            print("Total loss in epoch {} is {}.".format(epoch, tot_loss))
            epoch += 1
           
            
            model.eval()
            y_label = []
            y_pred = []
            test_batches = Batcher(val_X, val_y, self.device, batch_size=batch_size, max_iter=1)
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
                            #print(label_s)
                            #print(pred_s)
                            for j in range(len(pred_s)):
                            #    print(label_s[j])
                            #    print(pred_s[j])
                            #    print(torch.argmax(pred_s[j]))
                                pred_t = int(torch.argmax(pred_s[j]))
                                label_t = int(label_s[j])
                                y_pred.append(pred_t)
                                y_label.append(label_t)
                     
      
            #print(len(y_pred))
            #print(y_pred)
            #print(len(y_label))
            #print(y_label)
            scores = {}
            accuracy = accuracy_score(y_label, y_pred, normalize=True)
            scores['accuracy'] = accuracy
            recall = recall_score(y_label, y_pred, average='weighted')
            scores['recall'] = recall
            precision = precision_score(y_label, y_pred, average='weighted')
            scores['precision'] = precision
            f = f1_score(y_pred, y_label, average='weighted')
            scores['f1_score'] = f
            scores['loss'] = int(tot_loss)
        

            print('model:', model_name, 'accuracy:', accuracy, 'precision:', precision, 'recall:', recall, 'f1_score:', f)

        
            self.save_model(epoch, model, optimizer, tot_loss, scores, hyperparameters, model_name)
        pass
    
    
    def test(self, test_X, test_y, model_class, best_model_path):
        # Finish this function so that it loads a model, test is and print results.
        self.test_X = test_X
        self.test_y = test_y
        self.model_class = model_class
        self.best_model_path = best_model_path
        
        inputsize = list(test_X.shape)[2]  # = 7, number of features per word
        samplesize = list(test_X.shape)[1]
        outputsize = 5 # number of ner labels
        hiddensize = inputsize//2  # might change this, seems like a reasonable number..
        
        checkpoint = self.load_model(best_model_path)
        hyperparameters = checkpoint["hyperparamaters"] # misspelled to fit with already given save_model function
        batch_size = hyperparameters["batch_size"]
        epochs = hyperparameters["epochs"]
        n_layers = hyperparameters["number_layers"]
        model_name = hyperparameters["model_name"]
        model_state_dict = checkpoint["model_state_dict"]
        
        model = model_class(inputsize, hiddensize, outputsize, n_layers)
        model.load_state_dict(model_state_dict)
        model = model.to(self.device)
        
       
        
        batches = Batcher(test_X, test_y, self.device, batch_size=batch_size, max_iter=epochs)
        
        model.eval()
        y_label = []
        y_pred = []
        for batch in batches:
            for sentence, label in batch:
                pred = model(sentence.float(), self.device)
                for i in range(pred.shape[0]):
                    pred_s = pred[i]
                    label_s = label[i]
                    #print(pred_s.shape)
                    for j in range(len(pred_s)):
                        pred_t = int(torch.argmax(pred_s[j]))
                        label_t = int(label_s[j])
                        y_pred.append(pred_t)
                        y_label.append(label_t)
                        
        scores = {}
        accuracy = accuracy_score(y_label, y_pred, normalize=True)
        scores['accuracy'] = accuracy
        recall = recall_score(y_label, y_pred, average='weighted')
        scores['recall'] = recall
        precision = precision_score(y_label, y_pred, average='weighted')
        scores['precision'] = precision
        f = f1_score(y_pred, y_label, average='weighted')
        scores['f1_score'] = f
        print('model:', model_name, 'accuracy:', accuracy, 'precision:', precision, 'recall:', recall, 'f1_score:', f)
        pass
    


class Batcher:
    def __init__(self, X, y, device, batch_size=50, max_iter=None):
        self.X = X
        self.y = y
        self.device = device
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.curr_iter = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.curr_iter == self.max_iter:
            raise StopIteration
        permutation = torch.randperm(self.X.size()[0], device=self.device)
        permX = self.X[permutation]
        permy = self.y[permutation]
        splitX = torch.split(permX, self.batch_size)
        splity = torch.split(permy, self.batch_size)
        
        self.curr_iter += 1
        return zip(splitX, splity)