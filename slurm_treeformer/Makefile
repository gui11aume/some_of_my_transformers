OUTDIR= $(SCRATCH)/Projects/Treeformer

TOKENIZER= flamboyant_tokenizer.json
PRETRAINED= new_tiger_ddp.pt
DATA= filtered_hits_and_decoys.txt.gz
TRAINED= $(OUTDIR)/trained_octopus-tau-0.02.pt

$(TRAINED): $(TOKENIZER) $(PRETRAINED) $(DATA)
	srun python train_treeformer.py $^ $@
