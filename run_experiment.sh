
# Experiment runner script

python main.py \
        --epochs=50 \
        --epoch_step=[45] \
        --device_ids=[0] \
        --batch-size=4 \
        --G-lr=0.001 \
        --D-lr=0.1 \
        --B-lr=0.001 \
        --print-freq=200 \
        --save_model_path='./checkpoint' \
        --log_dir='./logs' \
        --data_type='allData' \
        --data_path='../datasets/AAPD/aapd2.csv' \
        --utilize_unlabeled_data=True \



