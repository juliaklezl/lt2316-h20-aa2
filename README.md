## LT2316 H20 Assignment A2 : Ner Classification

Name: *Julia Klezl* 

## Notes on Part 1.

I decided to implement a very simple network, consisting of 2 linear layers with a GRU layer in between, and a LogSoftmax layer to get the final prediction. I chose the GRU layer because reading up on the different options (RNN, GRU, LSTM), it seemed that RNN's are inferior to the other two due to problems with long-term memory (or rather lack thereof). GRU and LSTM appear to have comparable performance, but GRU is slightly less complex (which means more computationally efficient), so I chose that one. If I had the time to work on this assignment more in-depth, it would be interesting to compare networks with either layer with the same dataset and other hyperparameters. The input and output length are the number of features in my dataset and the number of labels respectively. The hidden size and number of layers are hyperparameters that will be set and tested in the third part of the assignment. 

## Notes on Part 2.
I decided to only get and return the checkpoint in the load function, and unpack it when needed in the test function. I find it easier to keep an overview that way. In my train function, I extract all information about hyperparameters from the dictionary, and use the Batcher from the horses and cats demo to shuffle and split the data. I realized later that there are quicker ways to do this pre-made in pytorch, but it's nice to see exactly what happens in the batcher like this. I use Adam or SGD optimizer, depending on what's chosen in the hyperparameter dictionary, and NLLLoss, since that works well together with the LogSoftmax output of my model. Then I train the model and evaluate it for every epoch. In the evaluation, I get the accuracy, precision, recall, and F1-Score, print them, and save them to the scores dictionary. The test function follows the same pattern as the evaluation. 


## Notes on Part 3.

The parameters I am playing around with in my hyperparameter sets are the learning rate, number of layers, optimizer, hidden size, batch size, and epochs, since these seem to be the most obvious parameters to tune in order to improve a model. I tried learning rates between 0.1 and 0.00001, number of layers between 1 and 20, Adam and SGD optimizers, hidden sizes between 4 and 100, batch sizes between 3 and 64, and numbers of epochs between 3 and 200. I didn't get very interesting results though, because my models often tend to get very high scores due to choosing only the label 0, or very low scores (<1%). I'm not sure whether this is simply due to bad features and/or very unbalanced data, or due to some mistake I made in my model or trainer. I get a decreasing loss when training, so the model clearly learns something, but apparently not what I want it to learn. I'm a bit confused though because when my scores are good, they are all high, not just the accuracy at the cost of precision or recall (which is what I would have expected when a model learns to only pick one of the labels). 

*Edit:* I changed the way my padding is labeled in a1, before padding simply got the label 0 (not an NER), now it gets an additional label 5, just for padding. This helps, I get more varied results and predictions, but the tendency is still there. 

For this reason, I did not learn a lot by trying these different settings and looking at the rather similar, sometimes almost identical parallel plots. Also, the graph is very hard to decipher, so even for the models that have different scores, it is hard to get information about them through the graph alone (so I went back to looking at the scores). I chose the f1-score for the plot, since it's a mix of the precision and recall scores and should give a balanced impression of the performance that way. 
Even though many of the settings I tried barely affected the performance, the graph did show that having a very high (20) or very low (1) number of layers and single-digit numbers of epochs cause it to drop, and that the Adam optimizer seems to be better-suited for this model than SGD. 


## Notes on Part 4.
As described above, I had several models with almost identical or identical scores, so I randomly picked one of the two best ones. It has a learning rate of 0.1, 5 layers, adam optimizer, a hidden size of 4, batch size of 32, and 20 epochs. 
In the validation, this model reached an accuracy of 96.71%, precision of 94.94%, recall of 96.71, and f1-score of 97.72%. 
On the unseen test set, it reaches an accuracy of 96.95%, precision of 95.34%, recall of 96.95%, and f1-score of 97.87%. So they are slightly (but neglectably) better in the test.


## Notes on Part Bonus.

I used the random sample function of a1 to find sentences for this. Which tokens can and cannot be separated depends on how tokenization was done, of course. So in my case, since I remove all punctuation between words, the system already has difficulties recognizing that something is a list of drugs (or groups, in this case), like in this phrase (I took only the relevant part of the sentence, since it's relatively long:

'when administered with ethyl alcohol phenothiazines barbiturates mao inhibitors and other antidepressants' \
 0&nbsp;&nbsp;&nbsp;&nbsp;0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0    3     3       1              1            1   1          0   0     1
 
 I'm chosing the BIO scheme, since it's very simple, but sufficient to solve such problems. I also read that it is the most commonly used in the industry:
 
 'when administered with ethyl alcohol phenothiazines barbiturates mao inhibitors and other antidepressants'
  O    O            O    B-3   B-3     B-1            B-1          B-1 I-1        O   O     B-1
  
This is useful to mark the boundaries of most tokens we're facing here, but does not solve the problem of compound words that are separated by other words, or even overlapping such as in the following example:

'potentiation occurs with ganglionic or peripheral adrenergic blocking drugs'
 0            0      0    1          0  1          1          1        1
 
 the BIO encoding might look somthing like this below, but is not able to capture the fact that this talks about ganglionic blocking drugs *or* peripheral adregenic blocking drugs, since it assumes that words that belong together always stand next to each other, and that one word can only be part of one compound. 
 
'potentiation occurs with ganglionic or peripheral adrenergic blocking drugs'
 O            O      O    B-1        O  B-1        I-1        I-1      I-1
 
More complex encoding schemes, such as linear CRF are able to deal with this better (at least with the discontinuous part), but using them would significantly increase the complexity for all sentences, so since these problematic types of tokens are rather rare in the dataset we're dealing with, I think I would still choose the basic BIO-encoding if I were to work on the tokenization/encoding of this data in more detail. 
  

