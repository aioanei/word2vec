import numpy as np

class Word2VecModel:
    def __init__(self, vocab_size, embedding_dim, lr=0.025, num_neg_samples=5):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.num_neg_samples = num_neg_samples
        # Initialize weights with small random values
        self.W1 = np.random.rand(vocab_size, embedding_dim) * 0.01
        self.W2 = np.random.rand(embedding_dim, vocab_size) * 0.01

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def get_negative_samples(self, exclude_indices):
        if isinstance(exclude_indices, int):
            exclude_indices = [exclude_indices]
        neg_samples = []
        while len(neg_samples) < self.num_neg_samples:
            neg_idx = np.random.randint(0, self.vocab_size)
            if neg_idx not in exclude_indices:
                neg_samples.append(neg_idx)
        return neg_samples

class CBOW(Word2VecModel):
    def forward(self, context_indices):
        # average the embeddings of the context words
        self.h = np.mean(self.W1[context_indices], axis=0).reshape(1, -1)
        return self.h  # no softmax needed anymore

    def backward(self, context_indices, target_idx):
        neg_samples = self.get_negative_samples(
            exclude_indices=context_indices + [target_idx]
        )

        total_loss = 0
        dL_dh = np.zeros((1, self.embedding_dim))

        # positive sample
        score_pos = self.sigmoid(np.dot(self.h, self.W2[:, target_idx]))
        w2_pos = self.W2[:, target_idx].copy()
        self.W2[:, target_idx] += self.lr * (1 - score_pos) * self.h.flatten()
        dL_dh += (1 - score_pos) * w2_pos
        total_loss -= np.log(score_pos + 1e-10)

        # negative samples
        for n_idx in neg_samples:
            score_neg = self.sigmoid(np.dot(self.h, self.W2[:, n_idx]))
            w2_neg = self.W2[:, n_idx].copy()
            self.W2[:, n_idx] -= self.lr * score_neg * self.h.flatten()
            dL_dh -= score_neg * w2_neg
            total_loss -= np.log(1 - score_neg + 1e-10)

        # update W1 for all context words
        grad_w1 = (self.lr * dL_dh.reshape(-1)) / len(context_indices)
        for idx in context_indices:
            self.W1[idx] += grad_w1

        return float(total_loss)
    
class SkipGram (Word2VecModel):
    def forward(self, target_idx):
        self.h = self.W1[target_idx].reshape(1, -1)
        return self.h  # no softmax needed anymore
    
    def backward(self, target_idx, context_indices):
        total_loss = 0
        dL_dh = np.zeros((1, self.embedding_dim))

        for c_idx in context_indices:
            neg_samples = self.get_negative_samples(
                exclude_indices=context_indices + [target_idx]
            )

            # positive sample
            score_pos = self.sigmoid(np.dot(self.h, self.W2[:, c_idx]))
            w2_pos = self.W2[:, c_idx].copy()
            self.W2[:, c_idx] += self.lr * (1 - score_pos) * self.h.flatten()
            dL_dh += (1 - score_pos) * w2_pos
            total_loss -= np.log(score_pos + 1e-10)

            # negative samples
            for n_idx in neg_samples:
                score_neg = self.sigmoid(np.dot(self.h, self.W2[:, n_idx]))
                w2_neg = self.W2[:, n_idx].copy()
                self.W2[:, n_idx] -= self.lr * score_neg * self.h.flatten()
                dL_dh -= score_neg * w2_neg
                total_loss -= np.log(1 - score_neg + 1e-10)

        self.W1[target_idx] += self.lr * dL_dh.flatten()
        return float(total_loss / len(context_indices))