import iterable
import pytorch_lightning as pl
import sys
import sentencepiece
import torch
import transformers
import json

# Batch size should be 8192.
BATCH_SIZE = 44
ACC = 46

N_STEPS = 500000
TRAIN_BATCHES = N_STEPS * ACC

WARMUP = int(0.1 * N_STEPS)
DECAY = N_STEPS - WARMUP

DEBUG = False

class TokenizerCollatorForLM:
   def __init__(self, tokenizer):
      self.tokenizer = tokenizer
      self.collator = transformers.DataCollatorForLanguageModeling(
         tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
      )

   def __call__(self, examples):
      tokenized = self.tokenizer(
          examples,
          return_attention_mask = False,
          return_token_type_ids = False,
          truncation = True,
          max_length = 512,
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


if __name__ == "__main__":

   pl.seed_everything(123)
   torch.set_float32_matmul_precision("medium")

   tokenizer_model_path   = "fr_tokenizer.model"
   train_data_path        = "remastered_oscar_2022.txt.gz"
   output_state_dict_path = "pretrained_model.pt"

   tokenizer = transformers.CamembertTokenizer(
       tokenizer_model_path,
   )

   # Standard BERT model.
   config = transformers.BertConfig(
       vocab_size = len(tokenizer),
       position_embedding_type = "relative_key_query",
   )
   model = transformers.BertForMaskedLM(config=config)

   harnessed_model = plTrainHarness(model, warmup=WARMUP, decay=DECAY).to(torch.bfloat16)

   train_data = iterable.IterableTextData(train_data_path, encoding="utf-8")

   data_loader = torch.utils.data.DataLoader(
       dataset = train_data,
       collate_fn = TokenizerCollatorForLM(tokenizer),
       batch_size = BATCH_SIZE,
       num_workers = 0 if DEBUG else 8,
       persistent_workers = False if DEBUG else True,
   )

   class EnhancedCheckpoint(pl.callbacks.ModelCheckpoint):
      def on_save_checkpoint(self, trainer, pl_module, checkpoint):
         state_dict = trainer.model.state_dict()
         state_dict["self.config"] = config
         torch.save(state_dict, "checkpoints/checkpointed.pt")

   save_checkpoint = EnhancedCheckpoint(
         dirpath = "checkpoints",
         every_n_train_steps = 512,
   )

   trainer = pl.Trainer(
         default_root_dir = "checkpoints",
         strategy = "ddp_find_unused_parameters_true",
#         strategy = pl.strategies.FSDPStrategy(
#            cpu_offload = torch.distributed.fsdp.CPUOffload(offload_params=True),
#            activation_checkpointing = [
#               transformers.models.bert.modeling_bert.BertLayer,
#            ],
#            mixed_precision = torch.distributed.fsdp.MixedPrecision(
#               param_dtype=torch.bfloat16,
#               reduce_dtype=torch.bfloat16,
#               buffer_dtype=torch.bfloat16,
#            ),
#            auto_wrap_policy = wrapping_policy
#         ),
         accelerator = "gpu",
         devices = 1 if DEBUG else torch.cuda.device_count(),
         max_epochs = -1,
         limit_train_batches = TRAIN_BATCHES,
         deterministic = False,
         # Options for a higher speed.
#         enable_progress_bar = False,
#         enable_model_summary = False,
#         logger = False,
         # Checkpointing.
         enable_checkpointing = True,
         callbacks = [save_checkpoint],
#         gradient_clip_val = 0.0,
         accumulate_grad_batches = ACC
   )

   trainer.fit(harnessed_model, data_loader)
   state_dict = model.state_dict()
   state_dict["self.config"] = config
   torch.save(state_dict, output_state_dict_path)
