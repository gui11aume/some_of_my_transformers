import re
import sys
import torch
import transformers

from encode_decode import EncoderDecoder

DEBUG = True


if __name__ == "__main__":

    torch.set_float32_matmul_precision("highest")

    pretrained_model_path = sys.argv[1]
 
 
    tokenizer_en = transformers.BertTokenizer.from_pretrained('bert-base-cased')
 
    tokenizer_fr = transformers.CamembertTokenizer.from_pretrained("camembert-base")
 
    # Load pretrained Bert (English).
    pretrained = torch.load(pretrained_model_path)
    en_config = pretrained.pop("en.config")
    fr_config = pretrained.pop("fr.config")
    en = transformers.BertModel(config=en_config, add_pooling_layer=False)
    fr = transformers.CamembertForCausalLM(config=fr_config)

    # Used only to load weights.
    model = EncoderDecoder(en, fr)
    print(model.load_state_dict(pretrained, strict=False))

    translator = transformers.EncoderDecoderModel(encoder=en, decoder=fr)
    translator = translator.to("cuda")

    generation_config = transformers.GenerationConfig(
#        do_sample = True, top_k = 512, top_p=.95, num_return_sequences = 1,
        do_sample = False, num_beams = 64, num_beam_groups = 4, early_stopping = False,
        pad_token_id = tokenizer_fr.pad_token_id,
        bos_token_id = tokenizer_fr.bos_token_id,
        eos_token_id = tokenizer_fr.eos_token_id,
#        repetition_penalty = 12.,
#        length_penalty = -12.,
        temperature = 0,
        decoder_start_token_id = tokenizer_fr.eos_token_id,
    )

    en_text = """The European Union (EU) is a supranational political and economic union of 27 member states that are located primarily in Europe."""

    model.eval()
    with torch.no_grad():
        encoder_inputs = tokenizer_en(en_text, return_tensors="pt", return_token_type_ids=False).to("cuda")
        en_len = len(encoder_inputs["input_ids"][0])
        generation_config.update(**{
            "max_new_tokens": int(1.8 * en_len),
            "min_new_tokens": int(0.8 * en_len),
        })
        translated = translator.generate(
            generation_config = generation_config,
            **encoder_inputs
        )
        print(en_text)
        print(tokenizer_fr.decode(translated[0], skip_special_tokens=True))

