import numpy as np
from typing import Literal

class UtilisationFunctions:
    @staticmethod
    def softmax(x, output=None, derivative=False):
        if derivative:
            batch_size, n_classes = output.shape
            jacobians = np.empty((batch_size, n_classes, n_classes))
            for i in range(batch_size):
                s = output[i].reshape(-1, 1)
                jacobians[i] = np.diagflat(s) - np.dot(s, s.T)
            return jacobians
        
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        softmax_x = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return softmax_x
    
    @staticmethod
    def sparse_categorical_entropy(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1.0)
        batch_size = y_pred.shape[0]
        return -np.log(y_pred[np.arange(batch_size), y_true.flatten()])
    
    @staticmethod
    def accuracy(y_true, y_pred):
        return np.argmax(y_pred) == np.argmax(y_true)
    
    @staticmethod
    def reLU(x, output=None, derivative=False):
        return np.maximum(0,x) if not derivative else UtilisationFunctions.reLUDeriv(x)
    
    @staticmethod
    def reLUDeriv(x):
        return np.where(x > 0, 1.0, 0.0)
    
    @staticmethod
    def xavier(input_units, output_units, counter_type: Literal["weight", "bias"]="weight"):
        if (counter_type == "weight"):
            limit = np.sqrt(6 / (input_units + output_units))
            return np.random.uniform(-limit, limit, (input_units, output_units))
        else:
            return np.zeros((output_units))
    
    @staticmethod
    def pad(x: np.ndarray, pad_num: int):
        if len(x.shape) == 3:
            C, H, W = x.shape
            new_arr = np.zeros((C, H + 2 * pad_num, W + 2 * pad_num))
            for c in range(C):
                for i in range(H):
                    for j in range(W):
                        new_arr[c, i + pad_num, j + pad_num] = x[c, i, j]
            return new_arr
        elif len(x.shape) == 2:
            H, W = x.shape
            new_arr = np.zeros((H + 2 * pad_num, W + 2 * pad_num))
            for i in range(H):
                for j in range(W):
                    new_arr[i + pad_num, j + pad_num] = x[i, j]
            return new_arr
        else:
            raise ValueError("Unsupported input shape for padding")
    
    @staticmethod
    def convMul(big: np.ndarray, smol: np.ndarray, stride: int):
        big_copy = np.array(big)
        smol_copy = np.array(smol)

        out_size = ((len(big) - len(smol)) // stride) + 1
        convRes = np.zeros((out_size, out_size))
        for i in range(len(convRes)):
            for j in range(len(convRes)):
                smoled_big = big_copy[i*stride:i*stride+len(smol), j*stride:j*stride+len(smol)]
                convRes[i][j] = np.sum(smoled_big * smol_copy)
        return convRes
    
    @staticmethod
    def maxArray(arr: np.ndarray):
        return np.max(arr)
    
    @staticmethod
    def avgArray(arr: np.ndarray):
        return np.mean(arr)
    
    @staticmethod
    def pooling(big: np.ndarray, smol_size: int, stride: int, method: Literal["max", "average"]="max"):
        big_copy = np.array(big)

        out_size = ((big_copy.shape[0] - smol_size) // stride) + 1
        # print(f"Size Detail: {big_copy.shape[0]}, {smol_size}, {stride}, {big.shape[0]}, {big.shape[1]}")
        poolRes = np.zeros((out_size, out_size))

        for i in range(out_size):
            for j in range(out_size):
                smoled_big = big_copy[i*stride:i*stride+smol_size, j*stride:j*stride+smol_size]
                if (method == "max"):
                    poolRes[i][j] = UtilisationFunctions.maxArray(smoled_big)
                else:
                    poolRes[i][j] = UtilisationFunctions.avgArray(smoled_big)
        return poolRes
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        return x / (1 - x)
    
    @staticmethod
    def tanh_derivative(x):
        return (1 - x ** 2)