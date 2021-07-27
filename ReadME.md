The model used was open sourced by Antonio Rubio: https://github.com/arubior/bilstm
He has instructions on on this GIthub page on how to train the model. Most of what we've customized pertains to the preprocessing/post-processing of inputs to run inferences on this model to generate outfits.


### What is the model?

The  outfit generator is made up of a BiLSTM model that takes in a sequence of image embeddings corresponding to the items in the outfit and uses trained weights to output a hidden state vector. A SoftMax function is applied to the matrix multiplication of the hidden state vector and outfit item candidates and the logits that the output corresponds to are the probabilities of each item being the next item in outfit. 

### What is it used for?

List of uses in our system

- Generating a set of outfits based an item that the user has added and the items in their closet.
- Generating a set of outfits based on items in the closet (no starting item).

### What data was it trained on?

Short description of the data the model was trained on

Polyvore dataset from [polyvore.com](http://polyvore.com) which is now no longer active

- a json file with a set of 17 000 outfits
- Jpeg images for each item in each outfit can be downloaded here: https://drive.google.com/file/d/0B4Eo9mft9jwoNm5WR3ltVkJWX0k/view?resourcekey=0-U-30d1POF7IlnAE5bzOzPA
