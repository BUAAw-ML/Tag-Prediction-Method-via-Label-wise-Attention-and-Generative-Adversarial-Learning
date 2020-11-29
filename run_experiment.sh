
# Experiment runner script

python main.py \
        --epochs=40 \
        --epoch_step=30 \
        --device_ids=0 \
        --batch-size=10 \
        --G-lr=0.001 \
        --D-lr=0.1 \
        --B-lr=0.001 \
        --print-freq=200 \
        --save_model_path='./checkpoint' \
        --log_dir='./logs' \
        --data_type='All' \
        --data_path='../datasets/EUR-Lex' \
        --use_previousData=0 \
        --method='MultiLabelMAP' \
        --overlength_handle='truncation' \

python main.py \
        --epochs=40 \
        --epoch_step=30 \
        --device_ids=0 \
        --batch-size=10 \
        --G-lr=0.001 \
        --D-lr=0.1 \
        --B-lr=0.001 \
        --print-freq=200 \
        --save_model_path='./checkpoint' \
        --log_dir='./logs' \
        --data_type='All' \
        --data_path='../datasets/EUR-Lex' \
        --use_previousData=0 \
        --method='semiGAN_MultiLabelMAP' \
        --overlength_handle='truncation' \

python main.py \
        --epochs=40 \
        --epoch_step=30 \
        --device_ids=0 \
        --batch-size=10 \
        --G-lr=0.001 \
        --D-lr=0.1 \
        --B-lr=0.001 \
        --print-freq=200 \
        --save_model_path='./checkpoint' \
        --log_dir='./logs' \
        --data_type='All' \
        --data_path='../datasets/EUR-Lex' \
        --use_previousData=0 \
        --method='MultiLabelMAP' \
        --overlength_handle='skip' \

python main.py \
        --epochs=40 \
        --epoch_step=30 \
        --device_ids=0 \
        --batch-size=10 \
        --G-lr=0.001 \
        --D-lr=0.1 \
        --B-lr=0.001 \
        --print-freq=200 \
        --save_model_path='./checkpoint' \
        --log_dir='./logs' \
        --data_type='All' \
        --data_path='../datasets/EUR-Lex' \
        --use_previousData=0 \
        --method='semiGAN_MultiLabelMAP' \
        --overlength_handle='skip' \

#truncation  skip
#All  TrainTest  TrainTestTextTag
#MultiLabelMAP semiGAN_MultiLabelMAP

#utilize_unlabeled_data、学习率、epoch_step

#batch-size：1，4，8，16

#苏州服务器上数据：
#../datasets/ProgrammerWeb/programweb-data.csv
#../datasets/AAPD/aapd2.csv
#../datasets/gan-bert
#../datasets/EUR-Lex

#614服务器上数据：
#../../datasets/multiClass_text_classification/news_group20/news_group20.csv
#../../datasets/multiLabel_text_classification/ProgrammerWeb/programweb-data.csv
#../../datasets/multiLabel_text_classification/EUR-Lex


