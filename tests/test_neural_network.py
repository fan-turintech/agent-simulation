import sys
import os
import pytest
import numpy as np

# Add parent directory to path to import simulation modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neural_network import NeuralNetwork
from config import Config

class TestNeuralNetwork:
    """Test cases for the neural network implementation"""
    
    def setup_method(self):
        """Set up test cases"""
        self.input_size = 5
        self.hidden_size = 4
        self.output_size = 2
        self.nn = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
    
    def test_initialization(self):
        """Test neural network initialization"""
        assert self.nn.input_size == self.input_size
        assert self.nn.hidden_size == self.hidden_size
        assert self.nn.output_size == self.output_size
        
        # Check weights shapes
        assert self.nn.weights_input_hidden.shape == (self.hidden_size, self.input_size)
        assert self.nn.weights_hidden_output.shape == (self.output_size, self.hidden_size)
        
        # Check weights are initialized within [-1, 1]
        assert np.all(self.nn.weights_input_hidden >= -1)
        assert np.all(self.nn.weights_input_hidden <= 1)
        assert np.all(self.nn.weights_hidden_output >= -1)
        assert np.all(self.nn.weights_hidden_output <= 1)
    
    def test_forward(self):
        """Test forward pass of neural network"""
        # Create test input
        test_input = np.random.random(self.input_size)
        
        # Forward pass
        output = self.nn.forward(test_input)
        
        # Check output shape and range
        assert output.shape == (self.output_size,)
        assert np.all(output >= 0)
        assert np.all(output <= 1)  # Since we use sigmoid activation
    
    def test_get_weights(self):
        """Test getting weights from network"""
        weights = self.nn.get_weights()
        assert len(weights) == 2
        assert weights[0] is self.nn.weights_input_hidden
        assert weights[1] is self.nn.weights_hidden_output
    
    def test_mutate(self):
        """Test mutation of neural network weights"""
        mutation_rate = 1.0  # Ensure all weights are mutated
        mutation_range = 0.5
        
        # Get original weights
        original_weights_input_hidden = self.nn.weights_input_hidden.copy()
        original_weights_hidden_output = self.nn.weights_hidden_output.copy()
        
        # Create mutated network
        mutated_nn = self.nn.mutate(mutation_rate, mutation_range)
        
        # Check mutated weights are different
        assert not np.array_equal(mutated_nn.weights_input_hidden, original_weights_input_hidden)
        assert not np.array_equal(mutated_nn.weights_hidden_output, original_weights_hidden_output)
        
        # Check the mutation is within the specified range
        max_diff_input_hidden = np.max(np.abs(mutated_nn.weights_input_hidden - original_weights_input_hidden))
        max_diff_hidden_output = np.max(np.abs(mutated_nn.weights_hidden_output - original_weights_hidden_output))
        
        assert max_diff_input_hidden <= mutation_range
        assert max_diff_hidden_output <= mutation_range
