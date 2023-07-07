import lightning
import pytorch_lightning as pl
import sys
import torch
import transformers

import iterable

DEBUG = False

BATCH_SIZE = 128
MAX_EPOCHS = 3

ACC = 8

WARMUP = 512
DECAY = int(2e6 * MAX_EPOCHS / BATCH_SIZE / ACC)


class EncoderDecoder(torch.nn.Module):
   def __init__(self, en, fr):
      super().__init__()
      self.en = en
      self.fr = fr

   def forward(self, encoder_tokens, decoder_tokens, batch_idx):
      encoder_outputs = self.en(**encoder_tokens)
      decoder_input_ids = transformers.models.encoder_decoder.modeling_encoder_decoder.shift_tokens_right(
            decoder_tokens["input_ids"],
            self.fr.config.pad_token_id,
            self.fr.config.decoder_start_token_id
      )
      decoder_outputs = self.fr(
            input_ids = decoder_input_ids,
            #attention_mask = decoder_tokens["attention_mask"],
            encoder_attention_mask = encoder_tokens["attention_mask"],
            encoder_hidden_states = encoder_outputs.last_hidden_state,
      )
      token_logits = decoder_outputs.logits
      loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
      target = decoder_tokens["input_ids"]
      target[decoder_tokens["attention_mask"] == 0] = -100
      decoder_outputs.loss = loss_fct(
            token_logits.reshape(-1, self.fr.config.vocab_size),
            target.view(-1)
      )
      return decoder_outputs.loss


class TrainHarness(pl.LightningModule):
    def __init__(self, model, warmup=50, decay=1000000):
        super().__init__()
        self.model = model
        self.warmup = warmup
        self.decay = decay

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=1e-4)
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
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([
              warmup,
              linear_decay
        ])

        # Tik-tok.
        optimizer.step()
        scheduler.step()

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        encoder_tokens, decoder_tokens = batch
        loss = self.model(encoder_tokens, decoder_tokens, batch_idx)
        (current_lr,) = self.lr_schedulers().get_last_lr()
        info = { "loss": loss, "lr": current_lr }
        self.log_dict(dictionary=info, on_step = True, prog_bar = True)
        return loss


class TokenizerCollator:
    def __init__(self, tokenizer_en, tokenizer_fr, max_length=128):
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr
        self.max_length = max_length

    def __call__(self, examples):
        encoder_tokens = self.tokenizer_en(
            [ex["en"] for ex in examples],
            return_token_type_ids = False,
            return_attention_mask = True,
            return_tensors = "pt",
            truncation = True,
            padding = True,
            max_length = self.max_length
        )
        decoder_tokens = self.tokenizer_fr(
            [ex["fr"] for ex in examples],
            return_token_type_ids = False,
            return_attention_mask = True,
            return_tensors = "pt",
            truncation = True,
            padding = True,
            max_length = self.max_length
        )
        return encoder_tokens, decoder_tokens


if __name__ == "__main__":

    pl.seed_everything(123)
    torch.set_float32_matmul_precision("high")
 
    train_data_path = sys.argv[1]
    trained_model_path = sys.argv[2]
 
    tokenizer_en = transformers.BertTokenizer.from_pretrained('bert-base-cased')
 
    tokenizer_fr = transformers.CamembertTokenizer.from_pretrained("camembert-base")
 
    model_en = transformers.BertModel.from_pretrained("bert-base-cased")
    en_config = model_en.config
 
    base_model_fr = transformers.CamembertModel.from_pretrained("camembert-base")
    fr_config = base_model_fr.config
    fr_config.update({
        "is_decoder": True,
        "add_cross_attention": True,
        "decoder_start_token_id": tokenizer_fr.eos_token_id,
    })
    model_fr = transformers.CamembertForCausalLM(config=fr_config)
    missing_keys = model_fr.roberta.load_state_dict(base_model_fr.state_dict(), strict=False)
    if DEBUG:
        print(missing_keys)
 
    model = EncoderDecoder(model_en, model_fr).to(torch.bfloat16)
 
    train_data = iterable.IterableJSONData(train_data_path)
 
    data_loader = torch.utils.data.DataLoader(
          dataset = train_data,
          collate_fn = TokenizerCollator(tokenizer_en, tokenizer_fr),
          batch_size = BATCH_SIZE,
          num_workers = 0 if DEBUG else 8,
          persistent_workers = False if DEBUG else True,
    )
 
 
    harnessed_model = TrainHarness(model, warmup=WARMUP, decay=DECAY)
 
    # Checkpoint and keep models every epoch. The default is only
    # one epoch so this is useless. Leaving it here in case the
    # number of epochs is increased.
    class EnhancedCheckpoint(pl.callbacks.ModelCheckpoint):
       def on_save_checkpoint(self, trainer, pl_module, checkpoint):
          state_dict = trainer.model.state_dict()
          state_dict["en.config"] = en_config
          state_dict["fr.config"] = fr_config
          torch.save(state_dict, "checkpoints/checkpointed_test.pt")
 
    save_checkpoint = EnhancedCheckpoint(
          dirpath = "checkpoints",
          every_n_train_steps = 512,
    )
 
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    wrapping_policy = ModuleWrapPolicy([
       transformers.models.bert.modeling_bert.BertLayer,
    ])
 
    trainer = pl.Trainer(
          default_root_dir = "checkpoints",
#          strategy = pl.strategies.FSDPStrategy(
##             cpu_offload = torch.distributed.fsdp.CPUOffload(offload_params=True),
#             activation_checkpointing = [
#                transformers.models.bert.modeling_bert.BertLayer,
#                transformers.models.bert.modeling_camembert.CamembertLayer,
#             ],
#             mixed_precision = torch.distributed.fsdp.MixedPrecision(
#                param_dtype=torch.bfloat16,
#                reduce_dtype=torch.bfloat16,
#                buffer_dtype=torch.bfloat16,
#             ),
#             auto_wrap_policy = wrapping_policy
#          ),
          strategy = pl.strategies.DDPStrategy(find_unused_parameters=True),
          accelerator = "gpu",
          devices = 1 if DEBUG else torch.cuda.device_count(),
          accumulate_grad_batches = ACC,
          max_epochs = MAX_EPOCHS,
          deterministic = False,
          # Options for a higher speed.
#          enable_progress_bar = False,
#          enable_model_summary = False,
#          logger = False,
          # Checkpointing.
          enable_checkpointing = True,
          callbacks = [save_checkpoint],
          gradient_clip_val = 1.0,
    )
 
    
    trainer.fit(harnessed_model, data_loader)
 
    trainer.save_checkpoint("trained_test")
    state_dict = model.state_dict()
    state_dict["en.config"] = en_config
    state_dict["fr.config"] = fr_config
    torch.save(state_dict, trained_model_path)
 
