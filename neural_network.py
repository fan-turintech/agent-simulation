import numpy as np
import random
from config import Config

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, weights=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        if weights is None:
            # Initialize weights with random values between -1 and 1
            self.weights_input_hidden = np.random.uniform(-1, 1, (hidden_size, input_size))
            self.weights_hidden_output = np.random.uniform(-1, 1, (output_size, hidden_size))
        else:
            self.weights_input_hidden = weights[0]
            self.weights_hidden_output = weights[1]
    
    def forward(self, inputs):
        # Convert inputs to numpy array
        inputs = np.array(inputs)
        
        # Calculate hidden layer activations
        hidden_inputs = np.dot(self.weights_input_hidden, inputs)
        hidden_outputs = self.sigmoid(hidden_inputs)
        
        # Calculate output layer activations
        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
        final_outputs = self.sigmoid(final_inputs)
        
        return final_outputs
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def get_weights(self):
        return [self.weights_input_hidden, self.weights_hidden_output]
    
    def mutate(self, mutation_rate=None, mutation_range=None):
        """Create a mutated copy of the neural network"""
        # Use default parameters if not specified
        if mutation_rate is None:
            mutation_rate = Config.MUTATION_RATE
        if mutation_range is None:
            mutation_range = Config.MUTATION_RANGE
            
        # Create a copy of current weights
        mutated_weights = [np.copy(self.weights_input_hidden), 
                          np.copy(self.weights_hidden_output)]
        
        # Mutate weights with probability mutation_rate
        for weight_matrix in mutated_weights:
            mask = np.random.random(weight_matrix.shape) < mutation_rate
            mutations = np.random.uniform(-mutation_range, mutation_range, weight_matrix.shape)
            weight_matrix[mask] += mutations[mask]
        
        # Return a new neural network with mutated weights
        return NeuralNetwork(self.input_size, self.hidden_size, self.output_size, mutated_weights)
