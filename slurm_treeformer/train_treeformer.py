import gzip
import os
import lightning.pytorch as pl
import sys
import torch
import transformers

import tree_collators
import treeformer

from typing import List

DEBUG = False

# Configuration for Titan RTX, 24 GB.
BATCH_SIZE = 48
ACC = 1

# The training data set re-uses each query-hit pair 10 times (with
# different decoys each time). If the number of epochs is increased,
# the decoys will be used multiple times as well.
N_EPOCHS = 2


def contrastive_loss_for_batch_of_quadruplets(X):
    # Unflatten batch along T to recover the initial quandruplets
    # as shown below in QHDD1 (N = T/4 x 4 x H).
    # query1, hit1, decoy1, second_decoy1
    # query2, hit2, decoy2, second_decoy1
    # ...      QHDD1        ...
    QHDD1 = torch.stack(torch.split(X, [4] * (len(X) // 4)))
    # Roll array QHDD1 along T/4 (QHDD2) and concatenate with
    # QHDD1 to create 4 unrelated negatives for every query.
    shift = int(torch.randint(1, QHDD1.shape[0], [1]))
    QHDD2 = torch.roll(QHDD1, shift, dims=0)
    QHDD2[:, 1] = torch.roll(QHDD2[:, 1], 1, dims=0)
    QHDD2[:, 2] = torch.roll(QHDD2[:, 2], 2, dims=0)
    QHDD2[:, 3] = torch.roll(QHDD2[:, 3], 2, dims=0)
    QHD_ = torch.cat([QHDD1, QHDD2], dim=1)
    # Cosine similarity between the queries and the
    # five other elements of the row (N x 8).
    cos = torch.bmm(
        torch.nn.functional.normalize(QHD_[:, :1, :], dim=-1),
        torch.nn.functional.normalize(QHD_[:, 1:, :], dim=-1).transpose(1, 2),
    ).squeeze()
    # Compute the contrastive loss in log space directly.
    tau = 0.02  # <- Temperature factor.
    losses = -cos[:, 0] / tau + torch.logsumexp(cos / tau, dim=-1)
    return losses


class plTrainHarness(pl.LightningModule):
    "A Lightning train harness with AdamW, warmup and linear decay."

    def __init__(self, model, lr=5e-5):
        super().__init__()
        self.model = model
        self.lr = lr

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=self.lr)

        n_steps = self.trainer.estimated_stepping_batches
        n_warmup_steps = int(0.1 * n_steps)
        n_decay_steps = int(0.9 * n_steps)

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=n_warmup_steps
        )
        decay = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.01, total_iters=n_decay_steps
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup, decay],
            milestones=[n_warmup_steps],
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        # The root nodes are the first token/tensors in each tree/ batch.
        roots = outputs.last_hidden_state[:, 0, :]  # T x H.
        all_losses = contrastive_loss_for_batch_of_quadruplets(roots)
        loss = all_losses.mean()
        (current_lr,) = self.lr_schedulers().get_last_lr()
        info = { "loss": loss, "lr": current_lr }
        self.log_dict(dictionary=info, on_step=True, prog_bar=True)
        return loss


def prepare_data_schedule(f, basedir) -> List[List[str]]:
    path = lambda s: os.path.join(basedir, s[:2], s[2:6], f"{s[6:]}.json.gz")
    schedule: List[List[str]] = []
    for line in f:
        items: List[str] = line.decode("ascii").rstrip().split(",")
        Q: str = path(items[0]) # Query
        H: str = path(items[1]) # Hit
        for i in range(10):
            D1: str = path(items[2*i+2]) # Decoy #1
            D2: str = path(items[2*i+3]) # Decoy #2
            schedule.append([Q,H,D1,D2])
    return schedule


def local_path(txt):
   return os.path.join(os.environ.get("SCRATCH"), txt)


class DumpStateDict(pl.callbacks.ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        model = trainer.model.module.model
        state_dict = { k:v.to("cpu") for (k,v) in model.state_dict().items() }
        state_dict["self.config"] = model.config
        torch.save(state_dict, local_path("checkpointed_Treeformer.pt"))


if __name__ == "__main__":
    pl.seed_everything(123)

    tokenizer_path = sys.argv[1]
    pretrained_LSTM_BERT_path = sys.argv[2]
    path_to_data_schedule = sys.argv[3]
    trained_treeformer_state_dict = sys.argv[4]

    basedir = os.path.join(local_path("flamboyant_trees"))

    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path, pad_token="[PAD]", mask_token="[MASK]"
    )

    # The Bert model is pretrained with a completely different
    # structure, but it makes training much better (increases
    # the recall at 1000 by ~9%).
    model = treeformer.Treeformer.load_from_file(
        path=pretrained_LSTM_BERT_path, strict=False
    )

    # Train harness with AdamW and a scheduler with warmup
    # followed by linear learning decay.
    harnessed_model = plTrainHarness(model)

    # Training dataset.
    with gzip.open(path_to_data_schedule) as f:
        data_schedule = prepare_data_schedule(f, basedir)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=data_schedule,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0 if DEBUG else 2,
        collate_fn=tree_collators.GroupedTreeCollator(tokenizer.pad_token_id, basedir),
        persistent_workers=False if DEBUG else True,
    )

    trainer = pl.Trainer(
        default_root_dir=local_path("Projects/Treeformer/checkpoints"),
        strategy=pl.strategies.DeepSpeedStrategy(
            stage=2,
            offload_optimizer=False,
            offload_parameters=False,
        ),
        accelerator="gpu",
        devices=1 if DEBUG else -1,
        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1)),
        accumulate_grad_batches = ACC,
        precision="16-mixed",
        max_epochs=N_EPOCHS,
        deterministic=True,
        callbacks=[DumpStateDict(every_n_train_steps=512)],

    )

    # Do the training.
    trainer.fit(harnessed_model, train_data_loader)

    # Save results.
    model.save_to_file(trained_treeformer_state_dict)
