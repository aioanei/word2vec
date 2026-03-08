import numpy as np

class CBOW:
    def __init__(self, v_size, d_dims, lr = 0.025):
        self.vocab_size = v_size
        self.embedding_dim = d_dims
        self.lr = lr
        # initialize weights with small random values
        self.W1 = np.random.rand(v_size, d_dims) * 0.01
        self.W2 = np.random.rand(d_dims, v_size) * 0.01
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    def forward(self, context_indices):
        # average the embeddings of the context words
        self.h = np.mean(self.W1[context_indices], axis=0).reshape(1, -1)
        # compute scores for all words in the vocabulary
        u = np.dot(self.h, self.W2)

        self.y_pred = self.softmax(u)
        return self.y_pred
    def backward(self, context_indices, target_idx):
        e = self.y_pred.copy()
        e[0][target_idx] -= 1  # error term
        # update W2
        dW2 = np.dot(self.h.T, e)
        dh = np.dot(e, self.W2.T)
        self.W2 -= self.lr * dW2

        # update W1
        grad_w1 = (self.lr * dh.reshape(-1)) / len(context_indices)
        for idx in context_indices:
            self.W1[idx] -= grad_w1

        loss = -np.log(self.y_pred[0][target_idx] + 1e-10)  # add small value to prevent log(0)
        return loss