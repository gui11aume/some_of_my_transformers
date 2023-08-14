import json
import pytorch_lightning as pl
import sys
import torch
import transformers

import iterable


DEBUG = False

N_GPUS = 16 # 4 nodes of 4 GPUs each.
N_ITEMS = 22_671_314 # Size of the data.
MAX_EPOCHS = 1
BATCH_SIZE = 128
ACC = 1

TRAIN_BATCHES = N_ITEMS * MAX_EPOCHS / BATCH_SIZE / N_GPUS
GRADIENT_STEPS = TRAIN_BATCHES / ACC

WARMUP = int(0.1 * GRADIENT_STEPS)
DECAY = GRADIENT_STEPS - WARMUP

class TokenizerCollatorForLM:
   def __init__(self, tokenizer):
      self.tokenizer = tokenizer
      self.collator = transformers.DataCollatorForLanguageModeling(
         tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
      )

   def __call__(self, examples):
      tokenized = self.tokenizer(
          examples,
          return_attention_mask = False, # Pad with the collator.
          return_token_type_ids = False,
          truncation = True,
          max_length = 256,
      )
      return self.collator(tokenized["input_ids"])


class plTrainHarness(pl.LightningModule):
   def __init__(self, model, warmup=50, decay=1000000):
      super().__init__()
      self.model = model
      self.warmup = warmup
      self.decay = decay

   def configure_optimizers(self):
      optimizer = torch.optim.AdamW(
          self.trainer.model.parameters(),
          lr = 6e-4,
          betas = (.9, .98)
      )
      warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor = 0.01,
            end_factor = 1.,
            total_iters = self.warmup)
      linear_decay = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor = 1.,
            end_factor = 0.01,
            total_iters = self.decay)
      scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer = optimizer,
            schedulers = [warmup, linear_decay],
            milestones = [self.warmup]
      )
      return  [optimizer], [{"scheduler": scheduler, "interval": "step"}]

   def training_step(self, batch, batch_idx):
      outputs = self.model(**batch)
      (current_lr,) = self.lr_schedulers().get_last_lr()
      self.log_dict(
          dictionary = { "loss": outputs.loss, "lr": current_lr },
          on_step = True,
          prog_bar = True
      )
      return outputs.loss


class DataDispatcher:
    def __init__(self, env):
        self.env = env
    def __call__(self):
        world_size = env.world_size()
        global_rank = env.global_rank()
        worker_info = torch.utils.data.get_worker_info()
        local_worker_rk = worker_info.id
        local_worker_nb = worker_info.num_workers
        worker_rk = global_rank * local_worker_nb + local_worker_rk
        worker_nb = world_size * local_worker_nb
        return worker_rk, worker_nb


if __name__ == "__main__":

    pl.seed_everything(123)
    torch.set_float32_matmul_precision("medium")

    tokenizer_path         = sys.argv[1]
    train_data_path        = sys.argv[2]
    output_state_dict_path = sys.argv[3]

    tokenizer = transformers.PreTrainedTokenizerFast(
          tokenizer_file = tokenizer_path,
          bos_token = "[CLS]",
          eos_token = "[SEP]",
          unk_token = "[UNK]",
          sep_token = "[SEP]",
          pad_token = "[PAD]",
          cls_token = "[CLS]",
          mask_token = "[MASK]"
    )

    # Standard Roberta model.
    config = transformers.RobertaConfig(vocab_size = len(tokenizer))
    model = transformers.RobertaForMaskedLM(config=config)

    harnessed_model = plTrainHarness(model, warmup=WARMUP, decay=DECAY).to(torch.bfloat16)

    train_data = iterable.IterableTextData(
        train_data_path,
        encoding = "utf-8",
        dist_env = "slurm",
    )

    data_loader = torch.utils.data.DataLoader(
        dataset = train_data,
        collate_fn = TokenizerCollatorForLM(tokenizer),
        batch_size = BATCH_SIZE,
        num_workers = 0 if DEBUG else 2,
        persistent_workers = False if DEBUG else True,
    )

    class EnhancedCheckpoint(pl.callbacks.ModelCheckpoint):
       def on_save_checkpoint(self, trainer, pl_module, checkpoint):
          state_dict = trainer.model.state_dict()
          state_dict["self.config"] = config
          torch.save(state_dict, "checkpointed.pt")

    save_checkpoint = EnhancedCheckpoint(
          dirpath = "checkpoints",
          every_n_train_steps = 512,
    )

    trainer = pl.Trainer(
          default_root_dir = "checkpoints",
          strategy = "ddp",
          num_nodes = 4,
#          strategy = "ddp_find_unused_parameters_true",
#          strategy = pl.strategies.FSDPStrategy(
##             cpu_offload = torch.distributed.fsdp.CPUOffload(offload_params=True),
#             activation_checkpointing = [
#                transformers.models.roberta.modeling_roberta.RobertaLayer,
#             ],
#             mixed_precision = torch.distributed.fsdp.MixedPrecision(
#                param_dtype=torch.bfloat16,
#                reduce_dtype=torch.bfloat16,
#                buffer_dtype=torch.bfloat16,
#             ),
##             auto_wrap_policy = wrapping_policy
#          ),
          accelerator = "gpu",
          devices = 1 if DEBUG else -1,
          max_epochs = MAX_EPOCHS,
#          limit_train_batches = TRAIN_BATCHES,
          deterministic = False,
          # Options for a higher speed.
#          enable_progress_bar = False,
#          enable_model_summary = False,
#          logger = False,
          # Checkpointing.
          enable_checkpointing = True,
          callbacks = [save_checkpoint],
#          gradient_clip_val = 0.0,
          accumulate_grad_batches = ACC
    )

    trainer.fit(harnessed_model, data_loader)
    state_dict = model.state_dict()
    state_dict["self.config"] = config
    torch.save(state_dict, output_state_dict_path)
