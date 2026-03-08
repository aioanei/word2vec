from data_utils import TextProcessor
from model import CBOW
import time
import numpy as np

def main():
    WINDOW = 3
    DIM = 100
    EPOCHS = 5
    LIMIT = 10000
    LR = 0.05

    proc = TextProcessor(WINDOW)
    text = proc.download_text(LIMIT)
    vocab_size = proc.build_vocab(text)
    print(f"Vocabulary size: {vocab_size}")
    # Print first 5 words
    for idx in range(min(5, vocab_size)):
        print(f"{idx}: {proc.idx2word[idx]}")

    model = CBOW(vocab_size, DIM, LR)
    start_time = time.time()
    print("Starting training")

    for epoch in range(EPOCHS):
        total_loss = 0
        pairs_count = 0

        for i, (context, target) in enumerate(proc.get_batches(text)):
            model.forward(context)
            loss = model.backward(context, target)
            total_loss += loss
            pairs_count += 1
            if (i + 1) % 10000 == 0:
                print(f"Epoch {epoch + 1}/{EPOCHS} | Processed {i + 1} batches")
        avg_loss = total_loss / pairs_count
        print(f"Epoch {epoch + 1}/{EPOCHS} | Average Loss: {avg_loss:.4f}")
    
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    np.save("embeddings.npy", model.W1)
    np.savez("metadata.npz", word2idx=proc.word2idx, idx2word=proc.idx2word)
    print("Embeddings and metadata saved.")

if __name__ == "__main__":
    main()