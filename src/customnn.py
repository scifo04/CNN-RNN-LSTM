import numpy as np
from utilfunc import UtilisationFunctions
from typing import Literal
from rmsnorm import RMSNorm
import keras

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

        if self.padding > 0:
            x = np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)))


        size = (len(x[0]) - self.kernel_size) // self.stride + 1
        out = np.zeros((self.out_channels, size, size))
        for i in range(len(self.kernel)):
            temp_out = np.zeros((size, size))
            for j in range(len(self.kernel[i])):
                temp_conv = UtilisationFunctions.convMul(x[j], self.kernel[i][j], self.stride)
                temp_out = temp_out + temp_conv
            temp_out += self.bias[i]
            out[i] = temp_out
        return out
    
    def backward(self, dLast: np.ndarray, learning_rate=0.005):
        dinput = np.zeros_like(self.last_image)
        dweight = np.zeros_like(self.kernel)
        dbias = np.zeros(self.out_channels)

        for oc in range(self.out_channels):
            dbias[oc] = np.sum(dLast[oc])
            for ic in range(self.in_channels):
                for i in range(dLast.shape[1]):
                    for j in range(dLast.shape[2]):
                        col_start = i * self.stride
                        col_end = col_start + self.kernel.shape[3]
                        row_start = j * self.stride
                        row_end = row_start + self.kernel.shape[2]
                        # print(f"{i}, {j}, {self.kernel.shape}")

                        if row_end > self.last_image.shape[1] or col_end > self.last_image.shape[2]:
                            # print("More than")
                            continue

                        last_image_slice = self.last_image[ic, row_start:row_end, col_start:col_end]
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
        height = x.shape[1]
        size = (height - self.kernel_size) // self.stride + 1
        # print(f"Size: {size}")
        out = np.zeros((x.shape[0], size, size))
        for i in range(x.shape[0]):
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
    
class Flatten:
    def __init__(self):
        self.last_shape = None

    def forward(self, x: np.ndarray):
        self.last_shape = x.shape
        reshaped = x.reshape(1, -1)
        # print("📦 Shape after Flatten:", x.shape)
        return reshaped

    def backward(self, dLast: np.ndarray):
        return dLast.reshape(self.last_shape)
    
class FullyConnected:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: Literal["reLU", "softmax"],
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
        
        self.grad_biases = np.sum(grad, axis=0, keepdims=True)

        if self.use_rmsnorm:
            grad = self.rmsnorm.backward(grad)        

        self.grad_weights = self.input.T @ grad
        return grad @ self.weights.T
    
class TextVectorization:
    def __init__(self, max_tokens: int, output_mode, output_sequence_length):
        self.vectorizer = keras.layers.TextVectorization(max_tokens, output_mode, output_sequence_length)

    def adapt(self, dataset):
        self.vectorizer.adapt(dataset)

class Embedded:
    def __init__(self, input_dim, output_dim, embeddings_initializer=Literal["uniform", "xavier"]):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = embeddings_initializer
        self.last_input = None
        self.learning_rate = 0.005
        if (self.embeddings_initializer == "uniform"):
            self.weight = np.random.uniform(-0.05, 0.05, size=(input_dim, output_dim))
        elif (self.embeddings_initializer == "xavier"):
            limit = np.sqrt(6 / (input_dim + output_dim))
            self.weight = np.random.uniform(-limit, limit, size=(input_dim, output_dim))

    def forward(self, x: np.ndarray):
        self.last_input = x
        embedded_vector = self.weight[x]
        return embedded_vector
    
    def backward(self, dLast: np.ndarray):
        for i in range(self.last_input.shape[0]):
            for j in range(self.last_input.shape[1]):
                token_idx = self.last_input[i][j]
                self.weight[token_idx] -= self.learning_rate * dLast[i][j]

class Dropout:
    def __init__(self, dropout_rate=0.3):
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training_status = True

    def forward(self, x):
        if self.training_status:
            self.mask = (np.random.rand(*x.shape) > self.dropout_rate).astype(np.float32)
            return (x*self.mask)/(1.0 - self.dropout_rate)
        else:
            return x
        
    def backward(self, dLast):
        return (dLast * self.mask) / (1.0 - self.dropout_rate)
    
class RNN:
    def __init__(self, unit, input_size, timestep, return_sequence, bidirectional=False):
        self.unit = unit
        self.input_size = input_size
        self.timestep = timestep
        self.bidirectional = bidirectional
        self.last_input = None
        self.return_sequence = return_sequence

        self.input_weights = UtilisationFunctions.xavier(input_size, unit, 'weight')
        self.hidden_weights = UtilisationFunctions.xavier(unit, unit, 'weight')

        self.bias = UtilisationFunctions.xavier(1, unit, 'bias')

        self.h_t = np.zeros((timestep, unit))

        if self.bidirectional:
            self.input_weights_reverse = UtilisationFunctions.xavier(input_size, unit, 'weight')
            self.hidden_weights_reverse = UtilisationFunctions.xavier(unit, unit, 'weight')

            self.bias_reverse = UtilisationFunctions.xavier(1, unit, 'bias')

            self.h_t_reverse = np.zeros((timestep, unit))

    def forward(self, x: np.ndarray):
        self.last_input = x
        h = np.zeros((self.timestep, self.unit))

        if self.bidirectional:
            h_reverse = np.zeros((self.timestep, self.unit))

        for t in range(self.timestep):
            h_used = np.zeros((self.unit)) if t == 0 else self.h_t[t-1]
            x_used = x[t]
            h[t] = np.tanh(x_used @ self.input_weights + h_used @ self.hidden_weights + self.bias)
            self.h_t[t] = h[t]

        if self.bidirectional:
            for t in reversed(range(self.timestep)):
                h_used_reverse = np.zeros((self.unit)) if t == self.timestep - 1 else self.h_t_reverse[t+1]
                x_used_reverse = x[t]
                h_reverse[t] = np.tanh(x_used_reverse @ self.input_weights_reverse + h_used_reverse @ self.hidden_weights_reverse + self.bias_reverse)
                self.h_t_reverse[t] = h_reverse[t]

        if self.return_sequence:
            if self.bidirectional:
                return np.concatenate([h, h_reverse], axis=1)
            else:
                return h
        else:
            if self.bidirectional:
                return np.concatenate([h[-1], h_reverse[0]])
            else:
                return h[-1]

class LSTM:
    def __init__(self, unit, input_size, timestep, return_sequence, bidirectional=False):
        self.unit = unit
        self.input_size = input_size
        self.timestep = timestep
        self.bidirectional = bidirectional
        self.last_input = None
        self.return_sequence = return_sequence

        self.input_weights = [UtilisationFunctions.xavier(input_size, unit, 'weight') for _ in range(4)]
        self.hidden_weights = [UtilisationFunctions.xavier(unit, unit, 'weight') for _ in range(4)]

        self.bias = [UtilisationFunctions.xavier(1, unit, 'bias') for _ in range(4)]

        self.h_t = np.zeros((timestep, unit))
        self.c_t = np.zeros((timestep, unit))

        if self.bidirectional:
            self.input_weights_reverse = [UtilisationFunctions.xavier(input_size, unit, 'weight') for _ in range(4)]
            self.hidden_weights_reverse = [UtilisationFunctions.xavier(unit, unit, 'weight') for _ in range(4)]

            self.bias_reverse = [UtilisationFunctions.xavier(1, unit, 'bias') for _ in range(4)]

            self.h_t_reverse = np.zeros((timestep, unit))
            self.c_t_reverse = np.zeros((timestep, unit))

    def forward(self, x: np.ndarray):
        self.last_input = x
        h = np.zeros((self.timestep, self.unit))
        c = np.zeros((self.timestep, self.unit))

        if self.bidirectional:
            h_reverse = np.zeros((self.timestep, self.unit))
            c_reverse = np.zeros((self.timestep, self.unit))

        for t in range(self.timestep):
            h_used = np.zeros((self.unit)) if t == 0 else self.h_t[t-1]
            c_used = np.zeros((self.unit)) if t == 0 else self.c_t[t-1]
            x_used = x[t]

            f = UtilisationFunctions.sigmoid(x_used @ self.input_weights[0] + h_used @ self.hidden_weights[0] + self.bias[0])
            i = UtilisationFunctions.sigmoid(x_used @ self.input_weights[1] + h_used @ self.hidden_weights[1] + self.bias[1])
            a = np.tanh(x_used @ self.input_weights[2] + h_used @ self.hidden_weights[2] + self.bias[2])
            o = UtilisationFunctions.sigmoid(x_used @ self.input_weights[3] + h_used @ self.hidden_weights[3] + self.bias[3])

            c[t] = (c_used * f) + (i * a)
            h[t] = np.tanh(c[t]) * o

            self.c_t[t] = c[t]
            self.h_t[t] = h[t]

        if self.bidirectional:
            for t in reversed(range(self.timestep)):
                h_used_reverse = np.zeros((self.unit)) if t == self.timestep - 1 else self.h_t_reverse[t+1]
                c_used_reverse = np.zeros((self.unit)) if t == self.timestep - 1 else self.c_t_reverse[t+1]
                x_used_reverse = x[t]

                f_reverse = UtilisationFunctions.sigmoid(x_used_reverse @ self.input_weights_reverse[0] + 
                                                         h_used_reverse @ self.hidden_weights_reverse[0] +
                                                         self.bias_reverse[0])
                i_reverse = UtilisationFunctions.sigmoid(x_used_reverse @ self.input_weights_reverse[1] +
                                                         h_used_reverse @ self.hidden_weights_reverse[1] +
                                                         self.bias_reverse[1])
                a_reverse = np.tanh(x_used_reverse @ self.input_weights_reverse[2] +
                                    h_used_reverse @ self.hidden_weights_reverse[2] +
                                    self.bias_reverse[2])
                o_reverse = UtilisationFunctions.sigmoid(x_used_reverse @ self.input_weights_reverse[3] +
                                                         h_used_reverse @ self.hidden_weights_reverse[3] +
                                                         self.bias_reverse[3])
                
                c_reverse[t] = (c_used_reverse * f_reverse) + (i_reverse * a_reverse)
                h_reverse[t] = np.tanh(c_reverse[t]) * o_reverse

                self.c_t_reverse[t] = c_reverse[t]
                self.h_t_reverse[t] = h_reverse[t]

        if self.return_sequence:
            if self.bidirectional:
                return np.concatenate([h, h_reverse], axis=1)
            else:
                return h
        else:
            if self.bidirectional:
                return np.concatenate([h[-1], h_reverse[0]])
            else:
                return h[-1]