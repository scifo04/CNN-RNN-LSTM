import numpy as np
from typing import Literal

class UtilisationFunctions:
    @staticmethod
    def softmax(x):
        expo = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return expo / np.sum(expo, axis=-1, keepdims=True)
    
    @staticmethod
    def sparse_categorical_entropy(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1e-15)
        return -np.log(y_pred[np.arange(len(y_pred), y_true)])
    
    @staticmethod
    def reLU(x):
        return x if x > 0 else 0
    
    @staticmethod
    def reLUDeriv(x):
        return 1 if x > 0 else 0
    
    @staticmethod
    def xavier(input_units, output_units, counter_type: Literal["weight", "bias"]="weight"):
        if (counter_type == "weight"):
            limit = np.sqrt(6 / (input_units + output_units))
            return np.random.uniform(-limit, limit, (input_units, output_units))
        else:
            return np.zeros((output_units))
    
    @staticmethod
    def pad(x: np.ndarray, pad_num: int):
        new_arr = np.zeros((len(x)+2*pad_num, len(x)+2*pad_num))
        for i in range(pad_num, len(new_arr)-pad_num):
            for j in range(pad_num, len(new_arr[i])-pad_num):
                print(i,j)
                new_arr[i][j] = x[i-pad_num][j-pad_num]
        return new_arr
    
    @staticmethod
    def convMul(big: np.ndarray, smol: np.ndarray, stride: int, bias=0.0):
        big_copy = np.array(big)
        smol_copy = np.array(smol)

        out_size = ((len(big) - len(smol)) // stride) + 1
        convRes = np.zeros((out_size, out_size))
        for i in range(len(convRes)):
            for j in range(len(convRes)):
                smoled_big = big_copy[i*stride:i*stride+len(smol), j*stride:j*stride+len(smol)]
                convRes[i][j] = np.sum(smoled_big * smol_copy) + bias[j]
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

        out_size = ((len(big) - smol_size) // 2) + 1
        poolRes = np.zeros((out_size, out_size))

        for i in range(len(poolRes)):
            for j in range(len(poolRes[i])):
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