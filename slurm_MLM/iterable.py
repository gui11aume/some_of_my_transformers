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

    def __init__(self, dist_env=None):
        super().__init__()
        self.world_size_handle, self.rank_handle = {
               "slurm": ("SLURM_NTASKS", "SLURM_PROCID")
        }.get(dist_env, ("WORLD_SIZE",   "LOCAL_RANK"))

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
        world_size = int(os.environ.get(self.world_size_handle))
        global_rank = int(os.environ.get(self.rank_handle))
        local_rank = worker_info.id
        local_num_workers = worker_info.num_workers
        # Assume that each process has the same number of local workers.
        worker_rk = global_rank * local_num_workers + local_rank
        worker_nb = world_size * local_num_workers
        return itertools.islice(self.iterator, worker_rk, None, worker_nb)


class IterableJSONData(IterableData):
    "Iterate over the lines of a JSON file and uncompress if needed."

    def __init__(self, data_path, train=True, **kwargs):
        super().__init__(**kwargs)
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

    def __init__(self, data_path, train=True, encoding="ascii", **kwargs):
        super().__init__(**kwargs)
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
