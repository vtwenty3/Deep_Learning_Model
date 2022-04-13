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


### Model 1 
Architecture:

![1](/img/1.png)
