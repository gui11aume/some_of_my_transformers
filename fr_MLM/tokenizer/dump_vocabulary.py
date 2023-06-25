import gzip
import io
import sentencepiece
import sys

if __name__ == "__main__":

    input_path = sys.argv[1]

    tokenizer = sentencepiece.SentencePieceProcessor(
        model_file=input_path
    )

    for i in range(tokenizer.piece_size()):
        print(f"{tokenizer.id_to_piece(i)}\t{tokenizer.get_score(i)}")
