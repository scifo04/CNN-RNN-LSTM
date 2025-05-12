import numpy as np
import torch
from utilfunc import UtilisationFunctions
from typing import Literal
from rmsnorm import RMSNorm

def flatten(x: np.ndarray):
    return x.flatten()

class Conv2D:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, bias=0.0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = np.full((out_channels), bias)
        self.kernel = np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.last_image = None

    def forward(self, x: np.ndarray):
        self.last_image = x
        size = (len(x[0]) - self.kernel_size) // self.stride + 1
        out = np.zeros((self.out_channels, size, size))
        for i in range(len(self.kernel)):
            temp_out = np.zeros((size, size))
            for j in range(len(self.kernel[i])):
                temp_conv = UtilisationFunctions.convMul(x[j], self.kernel[i][j], self.stride, self.bias)
                temp_out = temp_out + temp_conv
            out[i] = temp_out
        return out
    
    def backward(self, dLast: np.ndarray, learning_rate=0.005):
        dinput = np.zeros_like(self.last_image)
        dweight = np.zeros_like(self.kernel_size)
        dbias = np.zeros(self.out_channels)

        for oc in range(self.out_channels):
            dbias[oc] = np.sum(dLast[oc])
            for ic in range(self.in_channels):
                for i in range(dLast.shape[1]):
                    for j in range(dLast.shape[2]):
                        col_start = i * self.stride
                        col_end = col_start + self.kernel.shape[2]
                        row_start = j * self.stride
                        row_end = row_start + self.kernel.shape[3]

                        last_image_slice = self.last_image[ic, col_start:col_end, row_start:row_end]
                        dweight[oc, ic] += last_image_slice * dLast[oc, i, j]
                        dinput[ic, col_start:col_end, row_start:row_end] += self.kernel[oc, ic] * dLast[oc, i, j]
        
        self.kernel -= learning_rate*dweight
        return dinput

class Pooling:
    def __init__(self, kernel_size: int, stride: int, method: Literal["max", "average"]="max"):
        self.kernel_size = kernel_size
        self.stride = stride
        self.method = method
        self.last_image = None

    def forward(self, x: np.ndarray):
        self.last_image = x
        size = (len(x) - self.kernel_size) // self.stride + 1
        out = np.zeros((len(x), size, size))
        for i in range(len(x)):
            out[i] = UtilisationFunctions.pooling(x[i], self.kernel_size, self.stride, self.method)
        return out
    
    def backward(self, dLast: np.ndarray):
        in_channels = self.last_image.shape[0]
        hout, wout = dLast.shape[1], dLast.shape[2]
        dinput = np.zeros_like(self.last_image)

        for ic in range(in_channels):
            for i in range(hout):
                for j in range(wout):
                    col_start = i * self.stride
                    col_end = col_start + self.kernel_size
                    row_start = j * self.stride
                    row_end = row_start + self.kernel_size

                    last_image_slice = self.last_image[ic, col_start:col_end, row_start:row_end]

                    if (self.method == "max"):
                        maxe_index = np.unravel_index(np.argmax(last_image_slice), last_image_slice.shape)
                        dinput[ic, col_start:col_end, row_start:row_end][maxe_index] += dLast[ic, i, j]
                    else:
                        dinput[ic, col_start:col_end, row_start:row_end] += dLast[ic, i, j] / (self.kernel_size * self.kernel_size)

        return dinput
    
class ReLU:
    def __init__(self):
        self.reLU = np.vectorize(UtilisationFunctions.reLU)
        self.last_image = None

    def forward(self, x: np.ndarray):
        self.last_image = x
        return self.reLU(x)
    
    def backward(self, dLast: np.ndarray):
        dinput = dLast * (self.last_image > 0)
        return dinput
    
class FullyConnected:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: Literal["relu", "softmax"],
        weight_init: Literal["zero", "uniform", "normal", "he", "xavier"],
        lower: float,
        upper: float,
        mean: float,
        variance: float,
        seed,
        use_rmsnorm: bool=True
    ):
        self.activation_name = activation
        self.activation = getattr(UtilisationFunctions, activation)

        if seed is not None:
            np.random.seed(seed)

        if weight_init == "zero":
            self.weights = np.zeros((input_size, output_size))
            self.biases = np.zeros((1, output_size))
        elif weight_init == "uniform":
            self.weights = np.random.uniform(lower, upper, (input_size, output_size))
            self.biases = np.random.uniform(lower, upper, (1, output_size))
        elif weight_init == "normal":
            self.weights = np.random.normal(
                mean, np.sqrt(variance), (input_size, output_size)
            )
            self.biases = np.random.normal(mean, np.sqrt(variance), (1, output_size))
        elif weight_init == "he":
            self.weights = np.random.normal(
                0, np.sqrt(2 / input_size), (input_size, output_size)
            )
            self.biases = np.zeros((1, output_size))
        elif weight_init == "xavier":
            limit = np.sqrt(6 / (input_size + output_size))
            self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
            self.biases = np.zeros((1, output_size))
        else:
            self.weights = np.zeros((input_size, output_size))
            self.biases = np.zeros((1, output_size))

        self.input = None
        self.output = None
        self.z = None
        self.grad_weights = np.zeros((input_size, output_size))
        self.grad_biases = np.zeros((1, output_size))
        
        self.use_rmsnorm = use_rmsnorm
        if self.use_rmsnorm:
            self.rmsnorm = RMSNorm(output_size)
        else:
            self.rmsnorm = None

    def forward(self, x):
        self.input = x
        z = x @ self.weights
        if self.use_rmsnorm:
            z = self.rmsnorm.forward(z)
        z = z + self.biases
        self.z = z
        self.output = self.activation(z)
        return self.output
    
    def backward(self, grad_output):
        if self.activation_name == "softmax":
            jacobians = self.activation(self.z, self.output, derivative=True)
            grad_list = []
            for i in range(self.output.shape[0]):
                grad_i = np.dot(grad_output[i : i + 1], jacobians[i])
                grad_list.append(grad_i)
            grad = np.concatenate(grad_list, axis=0)
        else:
            activation_grad = self.activation(self.z, self.output, derivative=True)
            grad = grad_output * activation_grad
        
        # calculate the bias grad
        self.grad_biases = np.sum(grad, axis=0, keepdims=True)

        # calculate the weight grad and g grad if applicable
        if self.use_rmsnorm:
            grad = self.rmsnorm.backward(grad)        

        self.grad_weights = self.input.T @ grad
        return grad @ self.weights.T