# Character-based Language Modeling with Long short-term memory 

This is the template for the assignment of the Deep Learning and Neural Network course.

The implementation of the character based language model with Vanilla (Elman) RNN is shown, together with the template for the LSTM.

# Requirement.

Python 3.7 and Numpy are the only requirement. If you have any problem running on Windows, please tell me. 

I have tested the code for both Windows and Linux under Anaconda: https://www.anaconda.com/products/individual. I recommend this because the Intel Math Kernel Librariy (MKL) will be automatically installed via Anaconda.

# Running.

First, ensure that you are in the same directory with the python files and the "data" directory with the "input.txt" inside. 

For the vanilla RNN you can run two things:

## Training
- This code is provided as a simple implementation of the Elman RNN model, and how to generate character sequences from this model. You are encouraged to change the model hyper-parameters (model size, learning rate, batch size ...) to see their effects in training.

```
python elman_rnn.py train
```


## Gradent checking 
- Checking the gradient correctness. This step is normally important when implementing back-propagation. The idea of grad-check is actually very simple:

+ We need to know how to verify the correctness of the back-prop implementation.
+ In order to do that we rely on comparison with the gradients computed using numerical differentiation
+ For each weight in the network we will have to do the forward pass twice (one by increasing the weight by \delta, and one by decreasing the weight by \delta)
+ The difference between two forward passes gives us the gradient for that weight
+ (maybe the code will be self-explanationable)

```
python elman_rnn.py gradcheck
```

# Your LSTM implementation
I have already prepared the same template so that you can run your implementation. 

The gradcheck should be very similar and the implementation should be able to pass. Due to randomness, it is possible to have a couple of weights not passing the check. By running the check several times, a correct implementation should give us less than 5% of the weights passing the check. You should also use a small number of hidden neurons to run it quickly. The parameters that I provided in the Elman network are a bit too large for this purpose. 

Then you can implement sampling and see how it trains to generate new characters (It should generate much better than the Vanilla RNN). 

Note that: the template code and the RNN code is just a guide to make you have a fast start. You are encouraged to make any (change) if necessary, as long as the final work ends up with an LSTM network. For example, in the other assignment you can find the implementation for Adam which might help learning faster than Adagrad (but more expensive to compute). 
