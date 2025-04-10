import numpy as np
import random
from config import Config

class NeuralNetwork:
    """
    A simple feed-forward neural network implementation with one hidden layer.
    
    This class provides functionality for creating, using, and evolving a basic neural network
    with configurable input, hidden, and output layer sizes.
    
    Attributes:
        input_size (int): Number of input neurons
        hidden_size (int): Number of neurons in the hidden layer
        output_size (int): Number of output neurons
        weights_input_hidden (numpy.ndarray): Weight matrix connecting input to hidden layer
        weights_hidden_output (numpy.ndarray): Weight matrix connecting hidden to output layer
    """
    
    def __init__(self, input_size, hidden_size, output_size, weights=None):
        """
        Initialize a neural network with specified architecture.
        
        Args:
            input_size (int): Number of input neurons
            hidden_size (int): Number of neurons in the hidden layer
            output_size (int): Number of output neurons
            weights (list, optional): Pre-defined weights as [input_hidden_weights, hidden_output_weights].
                                     If None, random weights will be generated.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        if weights is None:
            # Initialize weights with random values between -1 and 1
            # Shape: (hidden_size, input_size) for input->hidden connections
            self.weights_input_hidden = np.random.uniform(-1, 1, (hidden_size, input_size))
            # Shape: (output_size, hidden_size) for hidden->output connections
            self.weights_hidden_output = np.random.uniform(-1, 1, (output_size, hidden_size))
        else:
            # Use provided weights
            self.weights_input_hidden = weights[0]
            self.weights_hidden_output = weights[1]
    
    def forward(self, inputs):
        """
        Perform forward propagation through the neural network.
        
        Takes input values, processes them through the hidden layer with sigmoid activation,
        then through the output layer with sigmoid activation.
        
        Args:
            inputs (list or numpy.ndarray): Input values for the neural network
            
        Returns:
            numpy.ndarray: Output values after forward propagation
        """
        # Convert inputs to numpy array for matrix operations
        inputs = np.array(inputs)
        
        # Calculate hidden layer activations:
        # 1. Matrix multiplication of weights and inputs
        # 2. Apply sigmoid activation function
        hidden_inputs = np.dot(self.weights_input_hidden, inputs)
        hidden_outputs = self.sigmoid(hidden_inputs)
        
        # Calculate output layer activations:
        # 1. Matrix multiplication of weights and hidden layer outputs
        # 2. Apply sigmoid activation function
        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
        final_outputs = self.sigmoid(final_inputs)
        
        return final_outputs
    
    def sigmoid(self, x):
        """
        Apply the sigmoid activation function element-wise.
        
        The sigmoid function transforms values to the range (0, 1) with an S-shaped curve,
        providing non-linear activation capabilities.
        
        Args:
            x (float or numpy.ndarray): Input value(s) to the sigmoid function
            
        Returns:
            float or numpy.ndarray: Transformed value(s) between 0 and 1
        """
        return 1 / (1 + np.exp(-x))
    
    def get_weights(self):
        """
        Get the current weights of the neural network.
        
        Returns:
            list: A list containing two weight matrices:
                 [weights_input_hidden, weights_hidden_output]
        """
        return [self.weights_input_hidden, self.weights_hidden_output]
    
    def mutate(self, mutation_rate=None, mutation_range=None):
        """
        Create a mutated copy of the neural network.
        
        This method creates a new neural network with slightly modified weights
        based on the mutation parameters. Only a subset of weights (determined by
        mutation_rate) will be modified by random values within the mutation_range.
        
        Args:
            mutation_rate (float, optional): Probability (0-1) of each weight being mutated.
                                           Defaults to value from Config.
            mutation_range (float, optional): Maximum amount by which a weight can be adjusted.
                                            Defaults to value from Config.
        
        Returns:
            NeuralNetwork: A new neural network instance with mutated weights
        """
        # Use default parameters from Config if not specified
        if mutation_rate is None:
            mutation_rate = Config.MUTATION_RATE
        if mutation_range is None:
            mutation_range = Config.MUTATION_RANGE
            
        # Create a copy of current weights to avoid modifying the original network
        mutated_weights = [np.copy(self.weights_input_hidden), 
                          np.copy(self.weights_hidden_output)]
        
        # For each weight matrix:
        # 1. Create a boolean mask where True means "mutate this weight"
        # 2. Generate random mutation values within the specified range
        # 3. Apply mutations only to the selected weights (where mask is True)
        for weight_matrix in mutated_weights:
            mask = np.random.random(weight_matrix.shape) < mutation_rate
            mutations = np.random.uniform(-mutation_range, mutation_range, weight_matrix.shape)
            weight_matrix[mask] += mutations[mask]
        
        # Return a new neural network with mutated weights
        return NeuralNetwork(self.input_size, self.hidden_size, self.output_size, mutated_weights)