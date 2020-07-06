# LSTM-Task 

This is the template for the assignment of the Deep Learning and Neural Network course.

The implementation of the character based language model with Vanilla RNN is shown, together with the template for the LSTM.

# Requirement.

Python 3.7 and Numpy are the only requirement. If you have any problem running on Windows, please tell me. 

# Running.

First, ensure that you are in the same directory with the python files and the "data" directory with the "input.txt" inside. 

For the vanilla RNN you can run two things:

- Training it to see the loss function and the samples being generated every 1000 steps. You can manually change the hyperparameters to play around with the code a little bit.

python elman_rnn.py train

- Check the gradient correctness. This step is normally important when implementing back-propagation. The idea of grad-check is actually very simple:

+ We need to know how to verify the correctness of the back-prop implementation.
+ In order to do that we rely on comparison with the gradients computed using numerical differentiation
+ For each weight in the network we will have to do the forward pass twice (one by increasing the weight by \delta, and one by decreasing the weight by \delta)
+ The difference between two forward passes gives us the gradient for that weight
+ (maybe the code will be self-explanationable)

python elman_rnn.py gradcheck

# Your LSTM implementation
I have already prepared the same template so that you can run your implementation. 

The gradcheck should be very similar and the implementation should be able to pass it (If too many warnings happen and the relative difference is > 0.01 then its probably incorrect). 

Then you can implement sampling and see how it trains to generate new characters (It should generate much better than the Vanilla RNN).

Note that: the template code and the RNN code is just a guide to make you have a fast start. You are encouraged to make any (change) if necessary, as long as the final work ends up with an LSTM network. 
