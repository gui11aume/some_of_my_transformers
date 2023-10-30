import gzip
import json
import os
import re
import torch

from typing import Any, Callable, Dict, List, Optional


MAX_LENGTH = 32


class TreeCollator:
    def __init__(self, pad_token_id:int, basedir:str, max_nodes: int = 128):
        self.pad_token_id = pad_token_id
        self.max_nodes = max_nodes

    def doc_from_path(self, path):
        with gzip.open(path) as f:
            doc = json.load(f)
        return doc

    def __call__(self, examples: List[Dict[str, Any]]):
        tokenized = [self.doc_from_path(path) for path in examples]
        tree_lengths = [len(t["depth"]) for t in tokenized]
        input_ids = [t["input_ids"] for t in tokenized]
        depths = [t["depth"] for t in tokenized]
        # The fields "row" and "col" are the sparse representation of the
        # attention mask.
        rows = [t["row"] for t in tokenized]
        cols = [t["col"] for t in tokenized]
        # Add "self indices" to build attention matrices.
        for i in range(len(tree_lengths)):
            rows[i] += list(range(tree_lengths[i]))
            cols[i] += list(range(tree_lengths[i]))

        # Assume that the root node has largest index in the tree.
        # The root is usually the node with index 0, but in this
        # data set the root had no index and it was put last.
        # Rotate node indices to put the root int first token.
        input_ids = [[x[-1]] + x[:-1] for x in input_ids]
        depths = [[d[-1]] + d[:-1] for d in depths]
        for i in range(len(tree_lengths)):
            rows[i] = [(r + 1) % tree_lengths[i] for r in rows[i]]
            cols[i] = [(c + 1) % tree_lengths[i] for c in cols[i]]

        # Cut the trees at a maximum of 'max_nodes'. In depth-first
        # indexing, this means removing the right-most children and their
        # descent; in breadth-first indexing, this means removing the
        # nodes at greatest depth. In other cases, it is impossible to
        # predict what this does.
        tree_lengths = [min(ln, self.max_nodes) for ln in tree_lengths]
        input_ids = [x[: self.max_nodes] for x in input_ids]
        depths = [d[: self.max_nodes] for d in depths]
        keep: Callable[[int, int], bool] = (
            lambda r, c: r < self.max_nodes and c < self.max_nodes
        )
        for i in range(len(tree_lengths)):
            rc = [(r, c) for r, c in zip(rows[i], cols[i]) if keep(r, c)]
            rows[i] = [r for r, c in rc]
            cols[i] = [c for r, c in rc]

        # The depths are used as 'position_ids'.
        position_ids = torch.nn.utils.rnn.pad_sequence(
            sequences=[torch.tensor(d) for d in depths], batch_first=True
        )

        # Pads are removed first thing in the training loop to create
        # packed sequence... This is seemingly wasteful, but doing 
        # otherwise crashes with error messages related to CPU / GPU 
        # comunication. Padding is really fast so here we go.
        input_lengths = [len(xx) for x in input_ids for xx in x]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            sequences=[torch.tensor(xx) for x in input_ids for xx in x],
            batch_first=True,
            padding_value=float(self.pad_token_id),
        )

        # Craft attention masks from sparse 'rows' / 'cols'. In this
        # type, a node sees all its descendents and itself (so the
        # root sees all the nodes).
        L = max(tree_lengths)
        assert L <= self.max_nodes
        # TODO:
        # - It is unclear what the type of rows, cols, rc, etc. is, please define.
        attention_mask_descent = torch.stack(
            [
                torch.sparse_coo_tensor(
                    (row, col),  # Coordinates to fill.
                    [1] * len(row),  # Fill with 1 (rest is 0).
                    (L, L),  # Dimension of the mask.
                ).to_dense()
                for col, row in zip(rows, cols)
            ]
        )

        # Tests show that the "deep" attention above gives better performance
        # on recovery with the USPTO data set. Restricting the attention to
        # children only reduces the recovery by about 2%.
        # Transform the attention masks so that the nodes see only their
        # children but not their grand children and beyond.
        # attention_mask = attention_mask_descent.clone()
        # for k in range(len(attention_mask)):
        #     mask = attention_mask[k]
        #     # Make sure that the nodes are sorted
        #     # either breadth-first or depth-first.
        #     assert torch.tril(mask, diagonal=-1).sum() == 0
        #     # Remove attention to grand-children and beyond.
        #     for i in range(L):
        #         if mask[:, i].sum() < 3:
        #             continue
        #         (idx,) = torch.where(mask[:, i] == 1)
        #         mask[idx[:-2], i] = 0

        batch: Dict[str, Any] = {
            # "id": [t["id"] for t in tokenized], # (discarded)
            "input_ids": input_ids,
            "tree_lengths": tree_lengths,
            "position_ids": position_ids,
            "input_lengths": input_lengths,
            "attention_mask": attention_mask_descent,
        }

        return batch


class GroupedTreeCollator(TreeCollator):
    def __call__(self, examples: List[List[Dict[str, Any]]]):
        # Flatten the list of examples.
        flat_examples = [path for ex in examples for path in ex]
        return super().__call__(flat_examples)


def recursively_set_depth_and_descent(depth: int, node) -> List[int]:
    # The depth of a node is the distance from the root,
    # and the descent is the collection of children,
    # grand children etc.
    descent: List[int] = [
        int(child_node.get_id()) for child_node in node.get_children()
    ]
    node.attrs["depth"] = depth
    node.attrs["descent"] = descent
    for child in node.get_children():
        descent = recursively_set_depth_and_descent(depth + 1, child)
        node.attrs["descent"] += descent
    return node.attrs["descent"]


def collect_attributes_from_tree(tree) -> Dict[str, Any]:
    # The root node does not have a number. The simplest fix is
    # to give it a number that is higher than existing ones.
    N = len(tree.nodes)
    # TODO: check if this should not be 'str(N)'.
    tree.nodes[str(N - 1)] = tree.root
    text = [tree.nodes[str(i)].attrs["text"] for i in range(N)]
    recursively_set_depth_and_descent(0, tree.root)
    depth: List[int] = [tree.nodes[str(i)].attrs.pop("depth") for i in range(N)]
    descent: List[List[int]] = [
        tree.nodes[str(i)].attrs.pop("descent") for i in range(N)
    ]
    row = [x for a in descent for x in a]
    col = [x for a in [[i] * len(descent[i]) for i in range(N)] for x in a]
    return {
        "text": text,
        "depth": depth,
        "row": row,
        "col": col,
    }
