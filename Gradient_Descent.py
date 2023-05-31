'''
This code implements the linear regregression in pure python(no numpy)
It uses mean square error for loss calculation
'''

gradient = 0
intercept = 0
iterations = 10000
learning_rate = 0.05


def calculate_output_array(inputs_array, gradient, intercept):
    result = []
    for i in inputs_array:
        result.append(gradient * i + intercept)
    return result

def calculate_cost(expected_values, given_values):
    a = []
    for i in range(len(given_values)):
        a.append((expected_values[i] - given_values[i]) ** 2)
    return (sum(a)/len(given_values))

def update_intercept(expected_values, given_values, b):
    a = []
    for i in range(len(given_values)):
        a.append((expected_values[i] - given_values[i]))
    part =  (-2/len(given_values) * sum(a))             #partial derivative of b with respect to output
    b -= learning_rate * part
    return b

def update_gradient(expected_values, given_values, input_array, m):
    a = []
    for i in range(len(input_array)):
        a.append(input_array[i] * (expected_values[i] - given_values[i]))
    part =  (-2/len(input_array) * sum(a))              #partial derivative of m with respect to output
    m -= learning_rate * part
    return m

#Test data
input_array = [1, 2, 3, 4, 5, 6]        #input array
output_array = [-1, 2, 5, 8, 11, 14]    #output array

for i in range(iterations):
    outputs = calculate_output_array(input_array, gradient, intercept)
    gradient = update_gradient(output_array, outputs, input_array, gradient)
    intercept = update_intercept(output_array, outputs, intercept)
    #print(calculate_cost(output_array, outputs))


print("Gradient:\t", gradient,"\nIntercept:\t", intercept, "\nLoss:\t", calculate_cost(output_array, outputs))
