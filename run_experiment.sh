
# Experiment runner script

SEQ_LEN="64"
BS="64"
LR="2e-5"
EPOCHS="3"
cur_dir="data"
LABEL_RATE="0.02"

python -u ganbert.py \
        --task_name=QC-fine \
        --label_rate=${LABEL_RATE} \
        --do_train=true \
        --do_eval=true \
        --do_predict=false \
        --data_dir=${cur_dir} \
        --vocab_file=$BERT_BASE_DIR/vocab.txt \
        --bert_config_file=$BERT_BASE_DIR/bert_config.json \
        --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
        --max_seq_length=${SEQ_LEN} \
        --train_batch_size=${BS} \
        --learning_rate=${LR} \
        --num_train_epochs=${EPOCHS} \
        --warmup_proportion=0.1 \
        --do_lower_case=false \
        --output_dir=ganbert_output_model

