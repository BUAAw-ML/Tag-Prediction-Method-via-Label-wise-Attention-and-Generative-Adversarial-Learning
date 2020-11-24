
# Experiment runner script

python main.py \
        --epochs=50 \
        --epoch_step=13 \
        --device_ids=0 \
        --batch-size=10 \
        --G-lr=0.001 \
        --D-lr=0.1 \
        --B-lr=0.01 \
        --print-freq=200 \
        --save_model_path='./checkpoint' \
        --log_dir='./logs' \
        --data_type='allData' \
        --data_path='../datasets/AAPD/aapd2.csv' \
        --utilize_unlabeled_data = 0 \


#utilize_unlabeled_data、学习率、epoch_step、无监督损失，

#batch-size：1，4，8，16

#苏州服务器上数据：
#../datasets/ProgrammerWeb/programweb-data.csv
#../datasets/AAPD/aapd2.csv
#../datasets/gan_bert

#614服务器上数据：
#../../datasets/multiClass_text_classification/news_group20/news_group20.csv
#../../datasets/multiLabel_text_classification/ProgrammerWeb/programweb-data.csv
#../../datasets/multiLabel_text_classification/EUR-Lex


