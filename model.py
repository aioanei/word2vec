import numpy as np

class Word2VecModel:
    def __init__(self, vocab_size, embedding_dim, lr=0.025):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = lr
        # Initialize weights with small random values
        self.W1 = np.random.rand(vocab_size, embedding_dim) * 0.01
        self.W2 = np.random.rand(embedding_dim, vocab_size) * 0.01

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

class CBOW (Word2VecModel):
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
    
class SkipGram (Word2VecModel):
    def forward(self, target_idx):
        # get the embedding of the target word
        self.h = self.W1[target_idx].reshape(1, -1)
        # compute scores for all words in the vocabulary
        u = np.dot(self.h, self.W2)

        self.y_pred = self.softmax(u)
        return self.y_pred
    def backward(self, target_idx, context_indices):
        total_loss = 0
        dL_dh = np.zeros((1, self.d_dims))
        for c_idx in context_indices:
            e = self.y_pred.copy()
            e[0, c_idx] -= 1
            self.W2 -= self.lr * np.dot(self.h.T, e)
            dL_dh += np.dot(e, self.W2.T)
            total_loss -= np.log(self.y_pred[0, c_idx] + 1e-10)
        self.W1[target_idx] -= self.lr * dL_dh.flatten()
        return total_loss / len(context_indices)