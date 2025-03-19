use_mux=0
do_finetune=1

./zipformer/finetune.py \
  --world-size 2 \
  --num-epochs 20 \
  --start-epoch 1 \
  --exp-dir zipformer/exp_spanish_finetune${do_finetune}_mux${use_mux} \
  --use-fp16 1 \
  --base-lr 0.003 \
  --bpe-model ~/asr-projects/models/icefall-asr-commonvoice-fr-pruned-transducer-stateless7-streaming-2023-04-02/data/lang_bpe_500/bpe.model \
  --do-finetune ${do_finetune} \
  --use-mux ${use_mux} \
  --master-port 13024 \
  --finetune-ckpt ~/asr-projects/models/icefall-asr-commonvoice-fr-pruned-transducer-stateless7-streaming-2023-04-02/exp/pretrained.pt \
  --max-duration 1000 \
  --train-cuts data/fbank/cuts_train_spanish.jsonl.gz \
  --valid-cuts data/fbank/cuts_valid_spanish.jsonl.gz
