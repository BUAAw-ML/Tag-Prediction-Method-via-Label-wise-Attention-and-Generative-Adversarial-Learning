# Experiment runner script

#According to the relevant literature, generative adversarial learning requires a smaller learning rate
#python main.py \
#        --experiment_no='zyc_GAN_MultiLabelMAP'  \
#        --epochs=100 \
#        --epoch_step=90 \
#        --device_ids=0 \
#        --batch-size=4 \
#        --G-lr=0.001 \
#        --D-lr=0.1 \
#        --B-lr=0.001 \
#        --data_type='TrainTest_pkl' \
#        --data_path='../datasets/Freecode' \
#        --use_previousData=0 \
#        --model_type='LABert' \
#        --method='GAN_MultiLabelMAP' \
#        --overlength_handle='truncation' \
#        --min_tagFrequence=0  \
#        --max_tagFrequence=999999  \
#        --intanceNum_limit=999999 \
#        --resume=''

python main.py \
        --experiment_no='wqb_MultiLabelMAP'  \
        --epochs=100 \
        --epoch_step=90 \
        --device_ids=0 \
        --batch-size=4 \
        --G-lr=0.001 \
        --D-lr=0.1 \
        --B-lr=0.001 \
        --data_type='TrainTest_pkl' \
        --data_path='../datasets/Freecode' \
        --use_previousData=0 \
        --model_type='LABert' \
        --method='MultiLabelMAP' \
        --overlength_handle='truncation' \
        --min_tagFrequence=0  \
        --max_tagFrequence=999999  \
        --intanceNum_limit=999999 \
        --resume=''

#model_type: MLPBert, LABert

#method: MultiLabelMAP GAN_MultiLabelMAP
#overlength_handle: truncation  skip

#data_type: TrainTest_pkl:
#../datasets/AAPD
#../datasets/Freecode
#../datasets/TREC-IS

#data_type: TrainTest_text:
#../datasets/stack-overflow2000




