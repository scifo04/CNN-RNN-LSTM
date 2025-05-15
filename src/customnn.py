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
    
class Flatten:
    def __init__(self):
        self.last_image = None

    def forward(self, x: np.ndarray):
        return x.flatten().reshape(1,-1)
    
    def backward(self, dLast: np.ndarray):
        return dLast.reshape(self.last_image)
    
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
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True):
        # Attribute Definition
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.last_input = None
        self.hidden_state = None
        self.hidden_state_reverse = None

        # Weight Utilisation
        self.ih_weights = UtilisationFunctions.xavier(self.input_size, self.hidden_size, "weight")
        self.hh_weights = [UtilisationFunctions.xavier(self.hidden_size, self.hidden_size, "weight") for _ in range(num_layers)]

        if self.bidirectional:
            self.ih_weights_reverse = UtilisationFunctions.xavier(self.input_size, self.hidden_size, "weight")
            self.hh_weights_reverse = [UtilisationFunctions.xavier(self.hidden_size, self.hidden_size, "weight") for _ in range(num_layers)]

        # Bias Utilisation
        self.ih_bias = UtilisationFunctions.xavier(self.input_size, self.hidden_size, "bias")
        self.hh_bias = [UtilisationFunctions.xavier(self.hidden_size, self.hidden_size, "bias") for _ in range(num_layers)]

        if self.bidirectional:
            self.ih_bias_reverse = UtilisationFunctions.xavier(self.input_size, self.hidden_size, "bias")
            self.hh_bias_reverse = [UtilisationFunctions.xavier(self.hidden_size, self.hidden_size, "bias") for _ in range(num_layers)]

    def forward(self, x: np.ndarray):
        self.last_input = x
        seq_length = x.shape[0]
        self.hidden_state = np.zeros((self.num_layers, seq_length, self.hidden_size))
        self.hidden_state_reverse = np.zeros((self.num_layers, seq_length, self.hidden_size))
        h_res = np.zeros(self.hidden_size)
        h_res_reverse = np.zeros(self.hidden_size)

        for layer in range(self.num_layers):
            outputs = []
            outputs_reverse = []
            
            layer_input = x if layer == 0 else np.stack(final_outputs, axis=0)

            if (layer <= 0):
                for t in range(seq_length):
                    x_t = layer_input[t]
                    h_res = np.tanh(np.dot(x_t, self.ih_weights) + self.ih_bias + np.dot(self.hh_weights[layer], h_res) + self.hh_bias[layer])
                    outputs.append(h_res)
                    self.hidden_state[layer][t] = h_res

                if (self.bidirectional):
                    for t in reversed(range(seq_length)):
                        x_t = layer_input[t]
                        h_res_reverse = np.tanh(np.dot(x_t, self.ih_weights_reverse) + self.ih_bias_reverse + np.dot(self.hh_weights_reverse[layer], h_res_reverse) + self.hh_bias_reverse[layer])
                        outputs_reverse.insert(0, h_res_reverse)
                        self.hidden_state_reverse[layer][t] = h_res_reverse
            
            else:
                for t in range(seq_length):
                    x_t = layer_input[t]
                    h_res = np.tanh(np.dot(x_t, self.hh_weights[layer-1]) + self.hh_bias[layer-1] + np.dot(self.hh_weights[layer], h_res) + self.hh_bias[layer])
                    outputs.append(h_res)
                    self.hidden_state[layer][t] = h_res

                if (self.bidirectional):
                    for t in reversed(range(seq_length)):
                        x_t = layer_input[t]
                        h_res_reverse = np.tanh(np.dot(x_t, self.hh_weights_reverse[layer-1]) + self.hh_bias_reverse[layer-1] + np.dot(self.hh_weights_reverse[layer], h_res_reverse) + self.hh_bias_reverse[layer])
                        outputs_reverse.insert(0, h_res_reverse)
                        self.hidden_state_reverse[layer][t] = h_res_reverse

            final_outputs = [np.concatenate([forward, backward]) for forward, backward in zip(outputs, outputs_reverse)] if self.bidirectional else outputs

            x = np.stack(final_outputs, axis=0)

        return x
    
    def backward(self, dLast: np.ndarray):
        seq_length = dLast.shape[0]
        dweight_ih = np.zeros_like(self.ih_weights)
        dweight_hh = np.zeros_like(self.hh_weights)
        dbias_ih = np.zeros_like(self.ih_bias)
        dbias_hh = np.zeros_like(self.hh_bias)

        if (self.bidirectional):
            dweight_ih_reverse = np.zeros_like(self.ih_weights_reverse)
            dweight_hh_reverse = np.zeros_like(self.hh_weights_reverse)
            dbias_ih_reverse = np.zeros_like(self.ih_bias_reverse)
            dbias_hh_reverse = np.zeros_like(self.hh_bias_reverse)

        dh_next = np.zeros((self.num_layers, self.hidden_size))
        dh_next_reverse = np.zeros((self.num_layers, self.hidden_size))

        for layer in reversed(range(self.num_layers)):
            for t in reversed(range(seq_length)):
                h_t = self.hidden_state[layer][t]
                h_prev = self.hidden_state[layer][t - 1] if t > 0 else np.zeros_like((self.hidden_size))
                x_t = self.last_input[t]

                dh = dLast[t] + dh_next[layer]

                dtanh = (1 - h_t ** 2) * (dh)
                
                if (layer == 0):
                    dweight_ih += np.outer(x_t, dtanh)
                    dbias_ih += dtanh
                dweight_hh[layer] += np.outer(h_prev, dtanh)
                dbias_hh[layer] += dtanh

                dh_next[layer] = np.dot(self.hh_weights[layer], dtanh)

            if (self.bidirectional):
                for t in range(seq_length):
                    h_t_reverse = self.hidden_state_reverse[layer][t]
                    h_prev_reverse = self.hidden_state_reverse[layer][t-1] if t > 0 else np.zeros_like(self.hidden_size)
                    x_t_reverse = self.last_input[t]

                    dh_reverse = dLast[t] + dh_next_reverse[layer]

                    dtanh_reverse = (1 - h_t_reverse ** 2) * dh_reverse

                    if (layer == 0):
                        dweight_ih_reverse += np.outer(x_t_reverse, dtanh_reverse)
                        dbias_ih_reverse += dtanh_reverse
                    dweight_hh_reverse[layer] += np.outer(h_prev_reverse, dtanh_reverse)
                    dbias_hh_reverse[layer] += dtanh_reverse

                    dh_next_reverse[layer] = np.dot(self.hh_weights_reverse[layer], dtanh_reverse)
        
        return {
            "dweight_ih": dweight_ih,
            "dweight_ih_reverse": dweight_ih_reverse if self.bidirectional else None,
            "dweight_hh": dweight_hh,
            "dweight_hh_reverse": dweight_hh_reverse if self.bidirectional else None,
            "dbias_ih": dbias_ih,
            "dbias_ih_reverse": dbias_ih_reverse if self.bidirectional else None,
            "dbias_hh": dbias_hh,
            "dbias_hh_reverse": dbias_hh_reverse if self.bidirectional else None
        }
    
    def learn(self, dLast: np.ndarray, learning_rate: float):
        dGotten = self.backward(dLast)
        self.ih_weights -= learning_rate * dGotten["dweight_ih"]
        self.hh_weights -= learning_rate * dGotten["dweight_hh"]
        self.ih_bias -= learning_rate * dGotten["dbias_ih"]
        self.hh_bias -= learning_rate * dGotten["dbias_hh"]

        if self.bidirectional:
            self.ih_weights_reverse -= learning_rate * dGotten["dweight_ih_reverse"]
            self.hh_weights_reverse -= learning_rate * dGotten["dweight_hh_reverse"]
            self.ih_bias_reverse -= learning_rate * dGotten["dbias_ih_reverse"]
            self.hh_bias_reverse -= learning_rate * dGotten["dbias_hh_reverse"]
    
class LSTM:
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True):
        # Attribute Definition
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.last_input = None
        self.hidden_state = None
        self.hidden_state_reverse = None
        self.cell_state = None
        self.cell_state_reverse = None
        self.input_gate = None
        self.input_gate_reverse = None
        self.forget_gate = None
        self.forget_gate_reverse = None
        self.gate = None
        self.gate_reverse = None
        self.output_gate = None
        self.output_gate_reverse = None

        # Weight Utilisation
        self.ih_weights = [UtilisationFunctions.xavier(self.input_size, self.hidden_size, "weight") for _ in range(4)]
        self.hh_weights = [[UtilisationFunctions.xavier(self.hidden_size, self.hidden_size, "weight") for _ in range(4)] for _ in range(num_layers)]
        self.gate_weights = [[UtilisationFunctions.xavier(self.hidden_size, self.hidden_size, "weight") for _ in range(4)] for _ in range(num_layers)]

        if self.bidirectional:
            self.ih_weights_reverse = [UtilisationFunctions.xavier(self.input_size, self.hidden_size, "weight") for _ in range(4)]
            self.hh_weights_reverse = [[UtilisationFunctions.xavier(self.hidden_size, self.hidden_size, "weight") for _ in range(4)] for _ in range(num_layers)]
            self.gate_weights_reverse = [[UtilisationFunctions.xavier(self.hidden_size, self.hidden_size, "weight") for _ in range(4)] for _ in range(num_layers)]

        # Bias Utilisation
        self.ih_bias = [UtilisationFunctions.xavier(self.input_size, self.hidden_size, "bias") for _ in range(4)]
        self.hh_bias = [[UtilisationFunctions.xavier(self.hidden_size, self.hidden_size, "bias") for _ in range(4)] for _ in range(num_layers)]

        if self.bidirectional:
            self.ih_bias_reverse = [UtilisationFunctions.xavier(self.input_size, self.hidden_size, "bias") for _ in range(4)]
            self.hh_bias_reverse = [[UtilisationFunctions.xavier(self.hidden_size, self.hidden_size, "bias") for _ in range(4)] for _ in range(num_layers)]

    def forward(self, x: np.ndarray, return_sequence=True):
        self.last_input = x
        seq_length = x.shape[0]
        self.hidden_state = np.zeros((self.num_layers, seq_length, self.hidden_size))
        self.hidden_state_reverse = np.zeros((self.num_layers, seq_length, self.hidden_size))
        self.cell_state = np.zeros((self.num_layers, seq_length, self.hidden_size))
        self.cell_state_reverse = np.zeros((self.num_layers, seq_length, self.hidden_size))
        self.input_gate = np.zeros((self.num_layers, seq_length, self.hidden_size))
        self.input_gate_reverse = np.zeros((self.num_layers, seq_length, self.hidden_size))
        self.forget_gate = np.zeros((self.num_layers, seq_length, self.hidden_size))
        self.forget_gate_reverse = np.zeros((self.num_layers, seq_length, self.hidden_size))
        self.gate = np.zeros((self.num_layers, seq_length, self.hidden_size))
        self.gate_reverse = np.zeros((self.num_layers, seq_length, self.hidden_size))
        self.output_gate = np.zeros((self.num_layers, seq_length, self.hidden_size))
        self.output_gate_reverse = np.zeros((self.num_layers, seq_length, self.hidden_size))

        for layer in range(self.num_layers):
            h_output = []
            h_output_reverse = []
            c_output = []
            c_output_reverse = []

            if (layer <= 0):
                for t in range(seq_length):
                    x_t = x[t]

                    self.forget_gate[layer][t] = UtilisationFunctions.sigmoid(np.dot(x_t, self.ih_weights[0]) + np.dot(self.hidden_state[layer][t], self.gate_weights[layer][t][0] + self.ih_bias[0]))
                    self.input_gate[layer][t] = UtilisationFunctions.sigmoid(np.dot(x_t, self.ih_weights[1]) + np.dot(self.hidden_state[layer][t], self.gate_weights[layer][t][1]) + self.ih_bias[1])
                    self.gate[layer][t] = np.tanh(np.dot(x_t, self.ih_weights[2]) + np.dot(self.hidden_state[layer][t], self.gate_weights[layer][t][2]) + self.ih_bias[2])
                    self.output_gate[layer][t] = UtilisationFunctions.sigmoid(np.dot(x_t, self.ih_weights[3]) + np.dot(self.hidden_state[layer][t], self.gate_weights[layer][t][3]) + self.ih_bias[3])

                    c_prev = self.cell_state[layer][t - 1] if t > 0 else np.zeros_like(self.hidden_size)

                    self.cell_state[layer][t] = self.input_gate[layer][t] * self.gate[layer][t] + self.forget_gate[layer][t] * c_prev
                    self.hidden_state[layer][t] = self.output_gate[layer][t] * np.tanh(self.cell_state[layer][t])
                    h_output.append(self.hidden_state[layer][t])
                    c_output.append(self.cell_state[layer][t])

                if (self.bidirectional):
                    for t in reversed(range(seq_length)):
                        x_t = x[t]

                        self.forget_gate_reverse[layer][t] = UtilisationFunctions.sigmoid(np.dot(x_t, self.ih_weights_reverse[0]) + np.dot(self.hidden_state_reverse[layer][t], self.gate_weights_reverse[layer][t][0]) + self.ih_bias_reverse[0])
                        self.input_gate_reverse[layer][t] = UtilisationFunctions.sigmoid(np.dot(x_t, self.ih_weights_reverse[1]) + np.dot(self.hidden_state_reverse[layer][t], self.gate_weights_reverse[layer][t][1]) + self.ih_bias_reverse[1])
                        self.gate_reverse[layer][t] = np.tanh(np.dot(x_t, self.ih_weights_reverse[2]) + np.dot(self.hidden_state_reverse[layer][t], self.gate_weights_reverse[layer][t][2]) + self.ih_bias_reverse[2])
                        self.output_gate_reverse[layer][t] = UtilisationFunctions.sigmoid(np.dot(x_t, self.ih_weights_reverse[3]) + np.dot(self.hidden_state_reverse[layer][t], self.gate_weights_reverse[layer][t][3]) + self.ih_bias_reverse[3])

                        c_prev_reverse = self.cell_state_reverse[layer][t + 1] if t < seq_length - 1 else np.zeros((self.hidden_size))

                        self.cell_state_reverse[layer][t] = self.input_gate_reverse[layer][t] * self.gate_reverse[layer][t] + self.forget_gate_reverse[layer][t] * c_prev_reverse
                        self.hidden_state_reverse[layer][t] = self.output_gate_reverse[layer][t] * np.tanh(self.cell_state_reverse[layer][t])
                        h_output_reverse.append(self.hidden_state_reverse[layer][t])
                        c_output_reverse.append(self.cell_state_reverse[layer][t])

            else:
                for t in range(seq_length):
                    x_t = x[t]

                    self.forget_gate[layer][t] = UtilisationFunctions.sigmoid(np.dot(x_t, self.hh_weights[layer-1][0]) + np.dot(self.hidden_state[layer][t], self.gate_weights[layer][t][0]) + self.hh_bias[layer-1][0])
                    self.input_gate[layer][t] = UtilisationFunctions.sigmoid(np.dot(x_t, self.hh_weights[layer-1][1]) + np.dot(self.hidden_state[layer][t], self.gate_weights[layer][t][1]) + self.hh_bias[layer-1][1])
                    self.gate[layer][t] = np.tanh(np.dot(x_t, self.hh_weights[layer-1][2]) + np.dot(self.hidden_state[layer][t], self.gate_weights[layer][t][2]) + self.hh_bias[layer-1][2])
                    self.output_gate[layer][t] = UtilisationFunctions.sigmoid(np.dot(x_t, self.hh_weights[layer-1][3]) + np.dot(self.hidden_state[layer][t], self.gate_weights[layer][t][3]) + self.hh_bias[layer-1][3])

                    c_prev = self.cell_state[layer][t - 1] if t > 0 else np.zeros_like(self.hidden_size)

                    self.cell_state[layer][t] = self.input_gate[layer][t] * self.gate[layer][t] + self.forget_gate[layer][t] * c_prev
                    self.hidden_state[layer][t] = self.output_gate[layer][t] * np.tanh(self.cell_state[layer][t])
                    h_output.append(self.hidden_state[layer][t])
                    c_output.append(self.cell_state[layer][t])

                if (self.bidirectional):
                    for t in reversed(range(seq_length)):
                        x_t = x[t]

                        self.forget_gate_reverse[layer][t] = UtilisationFunctions.sigmoid(np.dot(x_t, self.hh_weights_reverse[layer-1][0]) + np.dot(self.hidden_state_reverse[layer][t], self.gate_weights_reverse[layer][t][0]) + self.hh_bias_reverse[layer-1][0])
                        self.input_gate_reverse[layer][t] = UtilisationFunctions.sigmoid(np.dot(x_t, self.hh_weights_reverse[layer-1][1]) + np.dot(self.hidden_state_reverse[layer][t], self.gate_weights_reverse[layer][t][1]) + self.hh_bias_reverse[layer-1][1])
                        self.gate_reverse[layer][t] = np.tanh(np.dot(x_t, self.hh_weights_reverse[layer-1][2]) + np.dot(self.hidden_state_reverse[layer][t], self.gate_weights_reverse[layer][t][2]) + self.hh_bias_reverse[layer-1][2])
                        self.output_gate_reverse[layer][t] = UtilisationFunctions.sigmoid(np.dot(x_t, self.hh_weights_reverse[layer-1][3]) + np.dot(self.hidden_state_reverse[layer][t], self.gate_weights_reverse[layer][t][3]) + self.hh_bias_reverse[layer-1][3])

                        c_prev_reverse = self.cell_state_reverse[layer][t + 1] if t < seq_length - 1 else np.zeros((self.hidden_size))

                        self.cell_state_reverse[layer][t] = self.input_gate_reverse[layer][t] * self.gate_reverse[layer][t] + self.forget_gate_reverse[layer][t] * c_prev_reverse
                        self.hidden_state_reverse[layer][t] = self.output_gate_reverse[layer][t] * np.tanh(self.cell_state_reverse[layer][t])
                        h_output_reverse.append(self.hidden_state_reverse[layer][t])
                        c_output_reverse.append(self.cell_state_reverse[layer][t])

            final_outputs = [np.concatenate([forward, backward]) for forward, backward in zip(h_output, h_output_reverse)] if self.bidirectional else h_output

            x = np.stack(final_outputs, axis=0)

        if return_sequence:
            return x
        else:
            return x[-1]
        
    def backward(self, dLast: np.ndarray):
        seq_length = dLast.shape[0]

        dweight_ih = np.zeros_like(self.ih_weights)
        dweight_hh = np.zeros_like(self.hh_weights)
        dbias_ih = np.zeros_like(self.ih_bias)
        dbias_hh = np.zeros_like(self.hh_bias)

        dh_next = np.zeros((self.hidden_size))
        dc_next = np.zeros((self.hidden_size))

        if (self.bidirectional):
            dweight_ih_reverse = np.zeros_like(self.ih_weights_reverse)
            dweight_hh_reverse = np.zeros_like(self.hh_weights_reverse)
            dbias_ih_reverse = np.zeros_like(self.ih_bias_reverse)
            dbias_hh_reverse = np.zeros_like(self.hh_bias_reverse)

            dh_next_reverse = np.zeros((self.hidden_size))
            dc_next_reverse = np.zeros((self.hidden_size))

        for layer in range(self.num_layers):
            for t in reversed(range(seq_length)):
                i = self.input_gate[layer][t]
                f = self.forget_gate[layer][t]
                g = self.gate[layer][t]
                o = self.output_gate[layer][t]
                c = self.cell_state[layer][t]
                h = self.hidden_state[layer][t]
                x_t = self.last_input[layer][t]
                c_prev = self.cell_state[layer][t - 1]
                h_prev = self.hidden_state[layer][t - 1]

                dh = dLast[layer][t] + dh_next
                do = dh * np.tanh(c)
                do_act = do * UtilisationFunctions.sigmoid_derivative(o)
                
                dc = dh * o * UtilisationFunctions.tanh_derivative(np.tanh(c)) + dc_next
                di = dc * g
                di_act = di * UtilisationFunctions.sigmoid_derivative(i)

                dg = dc * i
                dg_act = dg * UtilisationFunctions.sigmoid_derivative(g)

                df = dc * c_prev
                df_act = df * UtilisationFunctions.sigmoid_derivative(f)

                dweight_ih[0] += np.outer(x_t, df_act)
                dweight_ih[1] += np.outer(x_t, di_act)
                dweight_ih[2] += np.outer(x_t, dg_act)
                dweight_ih[3] += np.outer(x_t, do_act)

                dweight_hh[0] += np.outer(h_prev, df_act)
                dweight_hh[1] += np.outer(h_prev, di_act)
                dweight_hh[2] += np.outer(h_prev, dg_act)
                dweight_hh[3] += np.outer(h_prev, do_act)

                dbias_ih[0] += df_act
                dbias_ih[1] += di_act
                dbias_ih[2] += dg_act
                dbias_ih[3] += do_act

                dbias_hh[0] += df_act
                dbias_hh[1] += di_act
                dbias_hh[2] += dg_act
                dbias_hh[3] += do_act

                dx = (np.dot(self.ih_weights[0], df_act) +
                    np.dot(self.ih_weights[1], di_act) +
                    np.dot(self.ih_weights[2], dg_act) +
                    np.dot(self.ih_weights[3], do_act))

                dh_next = (np.dot(self.hh_weights[layer][0], df_act) +
                        np.dot(self.hh_weights[layer][1], di_act) +
                        np.dot(self.hh_weights[layer][2], dg_act) +
                        np.dot(self.hh_weights[layer][3], do_act))

                dc_next = dc * f

            if (self.bidirectional):
                for t in range(seq_length):
                    i = self.input_gate_reverse[layer][t]
                    f = self.forget_gate_reverse[layer][t]
                    g = self.gate_reverse[layer][t]
                    o = self.output_gate_reverse[layer][t]
                    c = self.cell_state_reverse[layer][t]
                    h = self.hidden_state_reverse[layer][t]

                    x_t = self.last_input[layer][t]
                    c_next = self.cell_state_reverse[layer][t + 1] if t < seq_length - 1 else np.zeros(self.hidden_size)
                    h_next = self.hidden_state_reverse[layer][t + 1] if t < seq_length - 1 else np.zeros(self.hidden_size)

                    dh = dLast[t][self.hidden_size:] + dh_next_reverse
                    do = dh * np.tanh(c)
                    do_act = do * UtilisationFunctions.sigmoid_derivative(o)

                    dc = dh * o * UtilisationFunctions.tanh_derivative(np.tanh(c)) + dc_next_reverse
                    di = dc * g
                    di_act = di * UtilisationFunctions.sigmoid_derivative(i)

                    dg = dc * i
                    dg_act = dg * UtilisationFunctions.sigmoid_derivative(g)

                    df = dc * c_next
                    df_act = df * UtilisationFunctions.sigmoid_derivative(f)

                    dweight_ih_reverse[0] += np.outer(x_t, df_act)
                    dweight_ih_reverse[1] += np.outer(x_t, di_act)
                    dweight_ih_reverse[2] += np.outer(x_t, dg_act)
                    dweight_ih_reverse[3] += np.outer(x_t, do_act)

                    dweight_hh_reverse[0] += np.outer(h_next, df_act)
                    dweight_hh_reverse[1] += np.outer(h_next, di_act)
                    dweight_hh_reverse[2] += np.outer(h_next, dg_act)
                    dweight_hh_reverse[3] += np.outer(h_next, do_act)

                    dbias_ih_reverse[0] += df_act
                    dbias_ih_reverse[1] += di_act
                    dbias_ih_reverse[2] += dg_act
                    dbias_ih_reverse[3] += do_act

                    dbias_hh_reverse[0] += df_act
                    dbias_hh_reverse[1] += di_act
                    dbias_hh_reverse[2] += dg_act
                    dbias_hh_reverse[3] += do_act

                    dh_next_reverse = (
                        np.dot(self.hh_weights_reverse[layer - 1][0], df_act) +
                        np.dot(self.hh_weights_reverse[layer - 1][1], di_act) +
                        np.dot(self.hh_weights_reverse[layer - 1][2], dg_act) +
                        np.dot(self.hh_weights_reverse[layer - 1][3], do_act)
                    )

                    dc_next_reverse = dc * f

        return {
            "dweight_ih": dweight_ih,
            "dweight_ih_reverse": dweight_ih_reverse if self.bidirectional else None,
            "dweight_hh": dweight_hh,
            "dweight_hh_reverse": dweight_hh_reverse if self.bidirectional else None,
            "dbias_ih": dbias_ih,
            "dbias_ih_reverse": dbias_ih_reverse if self.bidirectional else None,
            "dbias_hh": dbias_hh,
            "dbias_hh_reverse": dbias_hh_reverse if self.bidirectional else None
        }
    
    def learn(self, dLast: np.ndarray, learning_rate: float):
        dGotten = self.backward(dLast)
        self.ih_weights -= learning_rate * dGotten["dweight_ih"]
        self.hh_weights -= learning_rate * dGotten["dweight_hh"]
        self.ih_bias -= learning_rate * dGotten["dbias_ih"]
        self.hh_bias -= learning_rate * dGotten["dbias_hh"]

        if self.bidirectional:
            self.ih_weights_reverse -= learning_rate * dGotten["dweight_ih_reverse"]
            self.hh_weights_reverse -= learning_rate * dGotten["dweight_hh_reverse"]
            self.ih_bias_reverse -= learning_rate * dGotten["dbias_ih_reverse"]
            self.hh_bias_reverse -= learning_rate * dGotten["dbias_hh_reverse"]
