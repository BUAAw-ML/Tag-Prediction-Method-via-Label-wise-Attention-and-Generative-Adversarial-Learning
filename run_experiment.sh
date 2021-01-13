
# Experiment runner script

#EN='02'  #experiment_no

#python main.py \
#        --experiment_no='Test'  \
#        --epochs=80 \
#        --epoch_step=70 \
#        --device_ids=0 \
#        --batch-size=4 \
#        --G-lr=0.001 \
#        --D-lr=0.01 \
#        --B-lr=0.001 \
#        --save_model_path='./checkpoint' \
#        --data_type='TrainTest_agNews' \
#        --data_path='../datasets/ag-news' \
#        --use_previousData=0 \
#        --method='semiGAN_MultiLabelMAP' \
#        --overlength_handle='skip' \
#        --min_tagFrequence=0  \
#        --max_tagFrequence=99999  \
#        --intanceNum_limit=99999 \
#        --data_split=0.0002  \
#        --test_description=''  \

python main.py \
        --experiment_no='zyz_GAN3_0<tag'  \
        --epochs=100 \
        --epoch_step=90 \
        --device_ids=0 \
        --batch-size=4 \
        --G-lr=0.001 \
        --D-lr=0.1 \
        --B-lr=0.001 \
        --save_model_path='./checkpoint' \
        --data_type='TrainTestTextTag' \
        --data_path='../datasets/RCV2' \
        --use_previousData=1 \
        --method='semiGAN_MultiLabelMAP' \
        --overlength_handle='truncation' \
        --min_tagFrequence=50  \
        --max_tagFrequence=100  \
        --intanceNum_limit=99999 \
        --data_split=200  \
        --test_description=''  \

#方法、epoch_step

#batch-size：1，4，8，16
#data_type: All  TrainTest  TrainTestTextTag
#method: MultiLabelMAP semiGAN_MultiLabelMAP
#overlength_handle: truncation  skip

#苏州服务器上数据：
#../datasets/ProgrammerWeb/programweb-data.csv  programweb-category  （TrainTest_programWeb）
#../datasets/AAPD  （TrainTest_programWeb）
#../datasets/gan-bert
#../datasets/EUR-Lex  (TrainTestTextTag)
#../datasets/RCV2  (TrainTestTextTag)
#../datasets/stack-overflow  (TrainTestTextTag)

#../datasets/ag-news

#614服务器上数据：
#../../datasets/multiClass_text_classification/news_group20/news_group20.csv
#../../datasets/multiLabel_text_classification/ProgrammerWeb/programweb-data.csv
#../../datasets/multiLabel_text_classification/EUR-Lex


