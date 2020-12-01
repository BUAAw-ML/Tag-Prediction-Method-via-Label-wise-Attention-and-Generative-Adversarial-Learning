
# Experiment runner script

EN='03'  #experiment_no

python main.py \
        --experiment_no=${EN}  \
        --epochs=30 \
        --epoch_step=15 \
        --device_ids=0 \
        --batch-size=8 \
        --G-lr=0.001 \
        --D-lr=0.1 \
        --B-lr=0.01 \
        --save_model_path='./checkpoint' \
        --data_type='TrainTestTextTag' \
        --data_path='../datasets/RCV2' \
        --use_previousData=0 \
        --method='MultiLabelMAP' \
        --overlength_handle='skip' \
        --min_tagFrequence=0  \
        --max_tagFrequence=30000  \
        --intanceNum_limit=15000 \
        --data_split=0.1  \

python main.py \
        --experiment_no=${EN}  \
        --epochs=45 \
        --epoch_step=40 \
        --device_ids=0 \
        --batch-size=8 \
        --G-lr=0.001 \
        --D-lr=0.1 \
        --B-lr=0.001 \
        --save_model_path='./checkpoint' \
        --data_type='TrainTestTextTag' \
        --data_path='../datasets/RCV2' \
        --use_previousData=0 \
        --method='semiGAN_MultiLabelMAP' \
        --overlength_handle='skip' \
        --min_tagFrequence=0  \
        --max_tagFrequence=30000  \
        --intanceNum_limit=15000 \
        --data_split=0.1  \




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


