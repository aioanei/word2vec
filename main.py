import data_utils

def main():
    proc = data_utils.TextProcessor()
    text = proc.download_text()
    vocab_size = proc.build_vocab(text)
    print(f"Vocabulary size: {vocab_size}")
    # Print first 5 words
    for idx in range(min(5, vocab_size)):
        print(f"{idx}: {proc.idx2word[idx]}")
    
if __name__ == "__main__":
    main()