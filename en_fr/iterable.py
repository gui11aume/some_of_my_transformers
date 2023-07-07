import itertools
import json
import gzip
import os
import torch


class IterableData(torch.utils.data.IterableDataset):
    """
    Defines the logic for iterable datasets (working over streams of
    data) in parallel multi-processing environments, e.g., multi-GPU.
    """

    @property
    def iterator(self):
        # Extend this class to define the stream.
        raise NotImplementedError

    def __iter__(self):
        # Get worker info if in multi-processing context.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.iterator

        # In multi-processing context, use 'os.environ' to
        # find global worker rank. Then use 'islice' to allocate
        # the items of the stream to the workers.
        process_rk = int(os.environ.get("LOCAL_RANK", 0))
        process_nb = int(os.environ.get("WORLD_SIZE", 1))
        local_worker_rk = worker_info.id
        local_worker_nb = worker_info.num_workers
        # Assume that each process has the same number of local workers.
        worker_rk = process_rk * local_worker_nb + local_worker_rk
        worker_nb = process_nb * local_worker_nb

        return itertools.islice(self.iterator, worker_rk, None, worker_nb)


class IterableJSONData(IterableData):
    "Iterate over the lines of a JSON file and uncompress if needed."

    def __init__(self, data_path, train=True):
        super().__init__()
        self.data_path = data_path
        self.train = train

    @property
    def iterator(self):
        "Define line-by-line iterator for json file."
        # Read the magic number.
        with open(self.data_path, "rb") as f:
            magic_number = f.read(2)
        # If file is gzipped, uncompress it on the fly.
        if magic_number == b'\x1f\x8b':
            iterator = map(
                    lambda line: json.loads(line.decode("ascii")),
                    gzip.open(self.data_path)
            )
        else:
            iterator = map(
                    lambda line: json.loads(line),
                    open(self.data_path)
            )
        return iterator


class IterableTextData(IterableData):
    "Iterate over the lines of a text file and uncompress if needed."

    def __init__(self, data_path, train=True, encoding="ascii"):
        super().__init__()
        self.data_path = data_path
        self.train = train
        self.encoding = encoding

    @property
    def iterator(self):
        "Define line-by-line iterator for text file."
        # Read the magic number.
        with open(self.data_path, "rb") as f:
            magic_number = f.read(2)
        # If file is gzipped, uncompress it on the fly.
        if magic_number == b'\x1f\x8b':
            iterator = map(
                    lambda line: line.decode(self.encoding),
                    gzip.open(self.data_path)
            )
        else:
            iterator = map(
                    lambda line: line,
                    open(self.data_path)
            )
        return iterator
