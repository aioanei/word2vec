# Word2Vec from Scratch

Implementation of Word2Vec (CBOW and Skip-Gram) using only NumPy, no ML frameworks.

## Architecture

Both models share a base class with two weight matrices:
- **W1** `(vocab_size × embed_dim)` — input embeddings
- **W2** `(embed_dim × vocab_size)` — output embeddings

Training uses negative sampling instead of softmax, updating only `K+1` weights per step instead of the entire vocabulary. From previous tests this method is 6 times faster and more precise.

### CBOW (Continuous Bag of Words)
Predicts the center word from surrounding context words. The context embeddings are averaged before computing the loss.

```
[context words] → average embedding → predict target
```

### Skip-Gram
Predicts context words from the center word.

```
[target word] → embedding → predict each context word
```

## Project Structure

```
word2vec/
├── main.py          # Training loop, comparison, evaluation
├── model.py         # CBOW and SkipGram implementations
├── data_utils.py    # Text download, vocab building, batch generation
├── evaluation.py    # Cosine similarity, nearest neighbors
```

## Usage

```bash
python main.py
```

Hyperparameters are set at the top of `main()` in `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `WINDOW` | 5 | Context window size |
| `DIM` | 100 | Embedding dimension |
| `EPOCHS` | 5 | Training epochs |
| `LIMIT` | 10000 | Max tokens to use from dataset |
| `LR` | 0.05 | Initial learning rate |

## Output

After training, the following files are saved:
- `embeddings_cbow.npy` — CBOW word vectors `(vocab_size × dim)`
- `embeddings_skipgram.npy` — Skip-Gram word vectors
- `metadata.npz` — `word2idx` and `idx2word` mappings

Evaluation is printed automatically:
- Nearest neighbors for a sample word
- Cosine similarity between word pairs

## Dataset

Uses the [text8](http://mattmahoney.net/dc/text8.zip) corpus (first 100M characters of Wikipedia). Downloaded automatically on first run.

## Requirements

```
numpy
```
