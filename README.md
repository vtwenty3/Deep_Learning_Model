# Santiment Analysis Deep Learning Model 

# Overview
This is relatively simple deep learning modedl, trained on 1500 movie reviews.  The representation technique chosen is Word Embeddings. The choice is motivated by the characteristics, flexibility and use cases of Word Embeddings. Moreover, working with word embeddings is more rewarding. The method is closer to Artificial intelligence and it is suitable for deep learning models. Its truly astonishing to observe how specific words are assigned with auto-generated features, which naturally capture the meaning of the word. Why not the traditional bag of words? 
## Bag Of Words model have many disadvantages over Word Embeddings:
1. Bag of Words model does not store any information related to sentence structure and word positioning.
2. Bag of Words cannot identify words with similar meanings based on context and positioning.
3. Bag of Words usually result in sparse vector, which can hinder performance and optimization.
4. Relays on hard-coded values which. 

## In contrast Word Embeddings Model:
1. Store the position of a word in leaned vector space (embedding).
2. Words used in a similar way have similar values, which naturally capture the meaning of the word, which results in a more accurate and flexible understanding of the text.
3. Can be more optimized, faster, and more accurate.
4. Using Machine Learning we can g enerate these vectors automatically.
5. Wider use cases - Autocompletion, summarization, voice recognition.


# Machine Learning Model
After extensive testing and experimenting with different Neural Network Architectures layers and approaches the final combination chosen is word embeddings and (BRNN) Bidirectional Recurrent Neural Networks. This approach is used with Long Short-Term Memory (LSTM) Recurrent Neural Networks. The choice is motivated by the simple implementation and testing. Moreover, this architecture is fully supported by Keras library. 

## Why RNN exactly? 
RNNs is superior compared to ANNs and CNNs when used for text classification, as CNNs power is in image and vide recognition. ANNs are not perfect match for NLP as they require too much computation and can get pretty complex overtime is the words are not hot encoded.
Adam optimizer proved to be better against rmsprop for 10 epochs. I assume that rmsprop could perform better with more epochs. I validated the results and i realised that the loss for the validation data is above 70%, so i changed the architecture to a simpler one, which gave slightly worse accuracy but better loss.

# Architecture:

I started with simple architecture and I slowly introduced more layers, this improved the accuracy, but overtime it increased the loss on evaluation data significantly. This side effect made me revert back to a simpler architecture consisting of only 3 layers.

-	Embedding Layer
-	Bidirectional LTSM Layer
-	Dense Layer

I am suppling 4 models with 4 different outcomes. The graphs showcase the strengths and weaknesses of different approaches. The architecture is similar on all of the models, but the difference is in hyperparameters (embedding dim, dropout rate, BRNN layer node count):


### Model 1 (Embedding Dim: 64, Nodes: 12, Droput: 0.6):
Architecture:

![1](/img/1.png)
Evaluation:

![2](/img/2.png)
Results:

![3](/img/3.png)

### Model 2 (Embedding Dim: 64 Embedding Dropout: 0.2 , Nodes: 24, Droput: 0.6)
Architecture:

![4](/img/4.png)
Evaluation:

![5](/img/5.png)
Results:

![6](/img/6.png)

### Model 3 (Embedding Dim: 32, Nodes: 12, Droput: 0.6)
Architecture:

![7](/img/7.png)
Evaluation:

![8](/img/8.png)
Results:

![9](/img/9.png)

### Model 4 (Embedding dim:32, Nodes: 16, Dropout: 0.6)
Architecture:

![10](/img/10.png)
Evaluation:

![11](/img/11.png)
Results:

![12](/img/12.png)


# Evaluation
For evaluation I have used real time monitoring of the learning progress of accu-racy and loss of the training and testing data. As addition after the training the result is evaluated with the evaluation data. So far the best result on evaluation data after 12 epochs is:

![14](/img/14.png)

This is achieved with the architecture of Model , the figure shows model7, but during the time of training and testing the architecture was identical to Model 1.

# Approach and Findings

Higher embedding dim seems to work better. I have done one more test, which involved increasing the epos for model3. This is the results:

![13](/img/13.png)

Number of epochs are highly influencial on the final result. Accuracy improves overtime, but the loss is getting higher with more epochs. That indicates that model gets overfitted. I have tried to counter that with different architectures and dropout rates, but the results are similar. 

Bag Of Words was the first approach tried, but it is obvious that this method is not appropriate for deep learning and is functional only in specific scenarios. Next word2vec library was used in attempt to train own vectors on the given train.csv file. Unfortunately, the results were pretty poor, as the data was not big enough for adequate training. Fasttext library developed by Facebook was an interesting find, but after two days trying to install it on google colab, Jupiter, and locally i gave up, as it was giving an error every time, no matter the platform or method used. The final decision was word embeddings with Keras, the implemen-tation is relatively straightforward and library is reliable and easy to import. Moreover, there is a strong community behind it, so more information can be found regarding tuning and eventual problems. Lots of combinations of different architectures and hyperparameters tried, but as far as my experience show, the simpler the architecture the better the results for the specific task.

There is simple pre-process which removes punctuation and removes uppercase, despite the tokenization part, which includes the removal of the uppercase. Its applied to all data (test, train, evaluation).


