# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 14:41:12 2017

@author: Administrator
"""

import copy, numpy as np
np.random.seed(0)
#import psyco; psyco.full()
from math import sqrt

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)


# training dataset generation
int2binary = {}
binary_dim = 8

largest_number = pow(2,binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]
    
def rwh_primes(n):   # Returns  a list of primes < n
    sieve = [True] * n
    for i in xrange(3,int(n**0.5)+1,2):
        if sieve[i]:
            sieve[i*i::2*i]=[False]*((n-i*i-1)/(2*i)+1)
    return [2] + [i for i in xrange(3,n,2) if sieve[i]]


# input variables
alpha = 0.1
input_dim = 2*binary_dim
hidden_dim = 16
output_dim = binary_dim


# initialize neural network weights
synapse_in = 2*np.random.random((input_dim,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,hidden_dim)) - 1
synapse_out = 2*np.random.random((hidden_dim,output_dim)) - 1


synapse_in_update = np.zeros_like(synapse_in)
synapse_1_update = np.zeros_like(synapse_1)
synapse_out_update = np.zeros_like(synapse_out)


# training logic
for j in range(50000):
    
    # generate a simple addition problem (a + b = c)
    a_int = np.random.randint(sqrt(largest_number)) # int version
    a = int2binary[a_int] # binary encoding

    b_int = np.random.randint(sqrt(largest_number)) # int version
    b = int2binary[b_int] # binary encoding

    # true answer
    c_int = a_int * b_int
    c = int2binary[c_int]
    
    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c)

    overallError = 0
    
    layer_out_deltas = list()
    layer_1_values = list()
    layer_2_values = list()
    layer_1_values.append(np.zeros(hidden_dim))
    layer_2_values.append(np.zeros(hidden_dim))
    

    # generate input and output
    X = np.concatenate((a,b))
    y = np.array(c).T

    # hidden layer
    layer_1 = sigmoid(np.dot(X,synapse_in))
    layer_2 = sigmoid(np.dot(layer_1,synapse_1))

    # output layer (new binary representation)
    layer_out = sigmoid(np.dot(layer_2,synapse_out))

    # did we miss?... if so, by how much?
    layer_out_error = y - layer_out
    layer_out_deltas = ((layer_out_error)*sigmoid_output_to_derivative(layer_out))
    overallError += np.abs(layer_out_error[0])

    # decode estimate so we can print it out
    d = (np.round(layer_out)).astype(int)
    
    # store hidden layer so we can use it in the next timestep
    #layer_1_values.append(copy.deepcopy(layer_1)) ## deepcopy so that when the values of layer_1 change the appended values don't also change. ie it doesn't point to the same data object
    #layer_2_values.append(copy.deepcopy(layer_2))
    
    
    #X = np.array([[a[position],b[position]]])
    #layer_1 = layer_1_values[-position-1]
    #layer_2 = layer_2_values[-position-1]
    
    # error at output layer
    #layer_out_delta = layer_out_deltas[-position-1]
    # error at hidden layer
    layer_2_delta = (layer_out_deltas.dot(synapse_out.T)) * sigmoid_output_to_derivative(layer_2)
    layer_1_delta = (layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)
    

    # let's update all our weights so we can try again
    synapse_1_update += np.atleast_2d(layer_1).dot(layer_2_delta)
    synapse_in_update += X.T.dot(layer_1_delta)
    synapse_out_update += np.atleast_2d(layer_2).T.dot([layer_out_deltas])
        
    
    synapse_in += synapse_in_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_out += synapse_out_update * alpha

    synapse_in_update *= 0
    synapse_1_update *= 0
    synapse_out_update *= 0
    
    # print out progress
    if(j % 1000 == 0):
        print "Error:" + str(overallError)
        print "Pred:" + str(d)
        print "True:" + str(c)
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print str(a_int) + " * " + str(b_int) + " = " + str(out)
        print "------------"
