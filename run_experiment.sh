
# Experiment runner script

#EN='02'  #experiment_no

python main.py \
        --experiment_no='tagFrequence>100_datasplit=0.05'  \
        --epochs=50 \
        --epoch_step=40 \
        --device_ids=0 \
        --batch-size=8 \
        --G-lr=0.001 \
        --D-lr=0.1 \
        --B-lr=0.01 \
        --save_model_path='./checkpoint' \
        --data_type='TrainTestTextTag' \
        --data_path='../datasets/EUR-Lex' \
        --use_previousData=0 \
        --method='MultiLabelMAP' \
        --overlength_handle='skip' \
        --min_tagFrequence=100  \
        --max_tagFrequence=20000  \
        --intanceNum_limit=10000 \
        --data_split=0.05  \

python main.py \
        --experiment_no='tagFrequence>100_datasplit=0.05'  \
        --epochs=80 \
        --epoch_step=70 \
        --device_ids=0 \
        --batch-size=8 \
        --G-lr=0.001 \
        --D-lr=0.1 \
        --B-lr=0.001 \
        --save_model_path='./checkpoint' \
        --data_type='TrainTestTextTag' \
        --data_path='../datasets/EUR-Lex' \
        --use_previousData=0 \
        --method='semiGAN_MultiLabelMAP' \
        --overlength_handle='skip' \
        --min_tagFrequence=100  \
        --max_tagFrequence=20000  \
        --intanceNum_limit=10000 \
        --data_split=0.05  \




#方法、学习率、epoch_step

#batch-size：1，4，8，16
#data_type: All  TrainTest  TrainTestTextTag
#method: MultiLabelMAP semiGAN_MultiLabelMAP
#overlength_handle: truncation  skip

#苏州服务器上数据：
#../datasets/ProgrammerWeb/programweb-data.csv
#../datasets/AAPD/aapd2.csv
#../datasets/gan-bert
#../datasets/EUR-Lex  (TrainTestTextTag)
#../datasets/RCV2  (TrainTestTextTag)

#614服务器上数据：
#../../datasets/multiClass_text_classification/news_group20/news_group20.csv
#../../datasets/multiLabel_text_classification/ProgrammerWeb/programweb-data.csv
#../../datasets/multiLabel_text_classification/EUR-Lex


