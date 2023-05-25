import numpy as np

#This creates a dense layer
class Layer:

    def __init__(self, input_neurons, output_nuerons):
        self.weights = 0.1 * np.random.randn(input_neurons, output_nuerons)
        self.biases = np.zeros((1, output_nuerons))
        
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

#This is ReLU activation class
class Activation_ReLU:
    def forward(self, input_array):
        self.output = np.maximum(0, input_array)

#This is SoftMax activation class
class Activation_SoftMax:
    def forward(self, input_array):
        expo_array = np.exp(input_array - np.max(input_array, axis=1, keepdims=True))
        normalized_array = expo_array / np.sum(expo_array, axis=1, keepdims=True)
        self.output = normalized_array

#Caterical Cross Entropy for Calculating the Loss
class Caterical_Cross_Entropy:
    def forward(self, output_array, expected_output):
        expected_output = np.array(expected_output)
        length = len(expected_output)
        clipped_output = np.clip(output_array, 1e-7, 1 - 1e-7)
        
        if(len(expected_output.shape) == 1):
            targeted_indices = clipped_output[range(length), expected_output]
        elif(len(expected_output.shape) == 2):
            targeted_indices = np.sum(clipped_output * expected_output, axis=1)

        losses = -np.log(targeted_indices)
        self.output = np.mean(losses)

#Measure of model accuracy
class Accuracy_Loss:
    def forward(self, y_pred, y_true):
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        outputs = np.argmax(y_pred, axis=1)

        #If y_true is one hot encoded
        if(len(y_true.shape) == 2):
            y_true = np.argmax(y_true, axis=1)
 
        self.output = np.mean(outputs == y_true)

#Calculate Loss Using Mean Square Error
class MeanSquareError_Loss:
    def forward(self, y_pred, y_true):
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)




#Test samples
X = [[1,2,3,4,5], [0,-2,-9,4.3,0.5], [-2, 3.3, -0.8, -1.6, 8]]
y1 = [[0,1], [0,1], [1,0]] #One-hot-encoded expected output
y2 = [1,1,0]        #Expected output



l1 = Layer(5, 3)
l2 = Layer(3, 4)
l3 = Layer(4, 2)
a1 = Activation_ReLU()
a2 = Activation_ReLU()
a3 = Activation_SoftMax()
loss1 = Caterical_Cross_Entropy()
loss2 = Caterical_Cross_Entropy()
acc = Accuracy_Loss()


l1.forward(X)
print(l1.output,"\n")
a1.forward(l1.output)
print(a1.output,"\n")
l2.forward(l1.output)
print(l2.output,"\n")
a2.forward(l2.output)
print(a2.output,"\n")
l3.forward(a2.output)
print(l3.output,"\n")
a3.forward(l3.output)
print(a3.output, '\n')
loss1.forward(a3.output, y1)
print(loss1.output)
loss2.forward(a3.output, y2)
print(loss2.output)
acc.forward(a3.output, y2)
print(acc.output)
acc.forward(a3.output, y1)
print(acc.output)
