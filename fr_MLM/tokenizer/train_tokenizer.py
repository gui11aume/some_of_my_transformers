import gzip
import io
import sentencepiece
import sys

if __name__ == "__main__":

    trained_model = io.BytesIO()

    with gzip.open("fr_wiki.txt.gz") as f:
        sentencepiece.SentencePieceTrainer.train(
           sentence_iterator = f,
           model_writer = trained_model,
           vocab_size = 32003,
           max_sentence_length = 32768,
           shuffle_input_sentence = True,
           character_coverage = 1.0,
           model_type = "unigram",
           train_extremely_large_corpus = True
        )

    with open("fr_tokenizer.model", "wb") as f:
        f.write(trained_model.getvalue())
