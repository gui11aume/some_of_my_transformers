from typing import Any, Dict

import torch
import transformers
import warnings


class Treeformer(torch.nn.Module):
    def __init__(self, embeddings=None, lstm=None, bert=None, config=None):
        super().__init__()

        # Instantiate from a (Bert) config file or from existing objects.
        if config is not None and bert is not None:
            raise Exception("Specify 'bert' or 'config' but not both.")

        self.config = bert.config if bert else config or transformers.BertConfig()

        # Check version.
        version = self.config.transformers_version
        if version is not None and version != transformers.__version__:
            warnings.warn(f"Treeformer instance was made with transformers {version} "
                    f"but you have {transformers.__version__}")
        self.config.transformers_version = transformers.__version__

        self.embeddings = embeddings or torch.nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.hidden_size,
        )

        self.lstm = lstm or torch.nn.LSTM(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size // 2,
            batch_first=True,
            bidirectional=True,
        )

        self.bert = bert or transformers.BertModel(config=config)

    def save_to_file(self, path):
        # Add 'self.config' to 'sate_dict' and save.
        state_dict = self.state_dict()
        state_dict["self.config"] = self.config
        torch.save(state_dict, path)

    @classmethod
    def load_from_file(cls, path, strict=True):
        state_dict = torch.load(path)
        config = state_dict.pop("self.config")

        embeddings = torch.nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        embeddings.load_state_dict(restrict_dict(state_dict, "embeddings."))

        lstm = torch.nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            batch_first=True,
            bidirectional=True,
        )
        lstm.load_state_dict(restrict_dict(state_dict, "lstm."))

        bert = transformers.BertModel(config)
        # Remove 'embeddings.position_ids' if present. It was removed
        # in a Transformers version after 4.23.1 where the Treeformer
        # was trained. It is not used anyway.
        # state_dict.pop("bert.embeddings.position_ids", None)
        bert.load_state_dict(
            restrict_dict(state_dict, "bert."),
            strict=strict,
        )

        return Treeformer(embeddings=embeddings, lstm=lstm, bert=bert)

    def reshape(self, x, lengths):
        # Assume that 'x' has dimensions B x H and that 'lengths'
        # is a list-like ojbect that contains the sizes of T trees.
        # Reshape 'x' to T x L x H, where L is 'max(lengths)'.
        # This requires adding rows to the trees with fewer than
        # L elements, so we add as many 0s as required.
        splitx = torch.split(x, lengths, dim=0)
        # Pad with 0s, but do not add rows for [CLS] and [SEP],
        # so that all tensors have dimensions L x H and stack for
        # final dimension T x L x H.
        L = max(lengths)
        reshaped = torch.stack(
            [torch.nn.functional.pad(b, (0, 0, 0, L - len(b))) for b in splitx]
        )
        return reshaped

    def forward(
        self, input_ids, position_ids, tree_lengths, input_lengths, attention_mask
    ):
        # The tensor 'input_ids' (B x L) consists of B tokenized
        # sentences, the longest with size L. The sentences
        # belong to T trees. First embed the tokens and obtain a
        # tensor of dimensions B x L x H.
        embeds = self.embeddings(input_ids)
        # Pack the B sentences and feed them to the LSTM.
        # NB: This changes the order of the sequences.
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embeds, lengths=input_lengths, batch_first=True, enforce_sorted=False
        )
        outputs, (h, c) = self.lstm(packed)
        # Collect history of 'h' vectors (fwd and rev) and max pool.
        # NB: This reverts the order of the sequences.
        hh, _ = torch.nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True, padding_value=float("-inf")
        )
        hh_max = torch.amax(hh, dim=1)
        # Each row of the tensor 'hh_max' (B x H) is the embedding
        # of a whole sentence. Each sentence is a token for the
        # Bert on top of the LSTM so we must pass T batches of
        # L embedded tokens.
        reshaped_hh_max = self.reshape(hh_max, tree_lengths)
        # The tensor 'reshaped_hh_max' (T x L x H) now matches the
        # 'position_ids' (T x L) and 'attention_mask' (T x L x L).
        return self.bert(
            inputs_embeds=reshaped_hh_max,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )


# Here-function to keep only the keys of a dictionary (d)
# that start with a given radix (rdx), and remove the radix.
def restrict_dict(d: Dict[str, Any], rdx: str):
    return {k.replace(rdx, "", 2): d[k] for k in d if k.startswith(rdx)}
