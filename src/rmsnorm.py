import numpy as np

class RMSNorm:
    def __init__(self, dim, eps=1e-8):
        self.dim = dim
        self.eps = eps
        self.g = np.ones(dim)
        self.grad_g = np.zeros_like(self.g)
        self.cache = {}

    def forward(self, x):
        self.cache['x'] = x
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        self.cache['rms'] = rms
        x_norm = x / rms
        self.cache['x_norm'] = x_norm
        out = x_norm * self.g
        return out
    
    def backward(self, grad_output):
        x = self.cache['x']
        x_norm = self.cache['x_norm']
        rms = self.cache['rms']
        dim = self.dim
        self.grad_g = np.sum(grad_output * x_norm, axis=0)
        grad_x_norm = grad_output * self.g
        sum_x_grad = np.sum(x * grad_x_norm, axis=-1, keepdims=True)
        dx = grad_x_norm / rms - (x * sum_x_grad) / (dim * rms**3)
        return dx