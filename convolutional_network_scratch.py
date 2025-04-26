import numpy as np

class ConvLayer:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / (filter_size * filter_size)
        
    def forward(self, input):
        self.input = input
        self.output = np.zeros((
            self.num_filters,
            input.shape[1] - self.filter_size + 1,
            input.shape[2] - self.filter_size + 1
        ))
        
        for i in range(self.num_filters):
            for y in range(self.output.shape[1]):
                for x in range(self.output.shape[2]):
                    self.output[i, y, x] = np.sum(
                        input[:, y:y+self.filter_size, x:x+self.filter_size] * self.filters[i]
                    )
        return self.output

class MaxPoolLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        
    def forward(self, input):
        self.input = input
        self.output = np.zeros((
            input.shape[0],
            input.shape[1] // self.pool_size,
            input.shape[2] // self.pool_size
        ))
        
        for i in range(input.shape[0]):
            for y in range(0, input.shape[1], self.pool_size):
                for x in range(0, input.shape[2], self.pool_size):
                    self.output[i, y//self.pool_size, x//self.pool_size] = np.max(
                        input[i, y:y+self.pool_size, x:x+self.pool_size]
                    )
        return self.output

class FCLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) / input_size
        self.bias = np.zeros(output_size)
        
    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias
        return self.output

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class CNN:
    def __init__(self):
        self.conv1 = ConvLayer(num_filters=8, filter_size=3)
        self.pool1 = MaxPoolLayer(pool_size=2)
        self.conv2 = ConvLayer(num_filters=16, filter_size=3)
        self.pool2 = MaxPoolLayer(pool_size=2)
        # Calculate the size after convolutions and pooling
        self.fc1 = FCLayer(16 * 5 * 5, 10)  # Assuming 28x28 input image
        
    def forward(self, x):
        # First convolution block
        x = self.conv1.forward(x)
        x = relu(x)
        x = self.pool1.forward(x)
        
        # Second convolution block
        x = self.conv2.forward(x)
        x = relu(x)
        x = self.pool2.forward(x)
        
        # Flatten and fully connected layer
        x = x.reshape(x.shape[0], -1)
        x = self.fc1.forward(x)
        return softmax(x)

# Example usage
if __name__ == "__main__":
    # Create a sample 28x28 image
    sample_image = np.random.randn(1, 28, 28)
    
    # Create and test the CNN
    cnn = CNN()
    output = cnn.forward(sample_image)
    
    print("Input shape:", sample_image.shape)
    print("Output shape:", output.shape)
    print("Output probabilities:", output)