from data_utils import TextProcessor
from model import CBOW, SkipGram
import time
import numpy as np

def train_cbow(vocab_size, dim, lr, epochs, proc, text):
    model = CBOW(vocab_size, dim, lr)
    print("\nTraining CBOW")
    start_time = time.time()
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0
        pairs_count = 0
        for i, (context, target) in enumerate(proc.get_batches(text)):
            model.forward(context)
            loss = model.backward(context, target)
            total_loss += loss
            pairs_count += 1
            if (i + 1) % 1000 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Processed {i + 1} batches")
        avg_loss = total_loss / pairs_count
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} | Average Loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"CBOW training time: {elapsed:.2f} seconds")
    return model, elapsed, epoch_losses

def train_skipgram(vocab_size, dim, lr, epochs, proc, text):
    model = SkipGram(vocab_size, dim, lr)
    print("\nTraining Skip-Gram")
    start_time = time.time()
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0
        pairs_count = 0
        for i, (context, target) in enumerate(proc.get_batches(text)):
            model.forward(target)
            loss = model.backward(target, context)
            total_loss += loss
            pairs_count += 1
            if (i + 1) % 1000 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Processed {i + 1} batches")
        avg_loss = total_loss / pairs_count
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} | Average Loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"Skip-Gram training time: {elapsed:.2f} seconds")
    return model, elapsed, epoch_losses

def main():
    WINDOW = 5
    DIM = 100
    EPOCHS = 5
    LIMIT = 10000  # limit number of tokens for faster training
    LR = 0.05

    proc = TextProcessor(WINDOW)
    text = proc.download_text(LIMIT)
    vocab_size = proc.build_vocab(text)
    print(f"Vocabulary size: {vocab_size}")
    for idx in range(min(5, vocab_size)):
        print(f"{idx}: {proc.idx2word[idx]}")

    cbow_model, cbow_time, cbow_losses = train_cbow(vocab_size, DIM, LR, EPOCHS, proc, text)
    sg_model, sg_time, sg_losses = train_skipgram(vocab_size, DIM, LR, EPOCHS, proc, text)

    print("\nComparison")
    print(f"CBOW Training time: {cbow_time:.2f}s Final Avg Loss: {cbow_losses[-1]:.4f} (Window: {WINDOW}, Dim: {DIM}, Epochs: {EPOCHS})")
    print(f"Skip-Gram Training time: {sg_time:.2f}s Final Avg Loss: {sg_losses[-1]:.4f} (Window: {WINDOW}, Dim: {DIM}, Epochs: {EPOCHS})")

    np.save("embeddings_cbow.npy", cbow_model.W1)
    np.save("embeddings_skipgram.npy", sg_model.W1)
    np.savez("metadata.npz", word2idx=proc.word2idx, idx2word=proc.idx2word)
    print("\nEmbeddings and metadata saved.")

if __name__ == "__main__":
    main()