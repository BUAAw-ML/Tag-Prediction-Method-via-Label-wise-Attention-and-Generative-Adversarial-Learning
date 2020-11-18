# Experimental progress and records

实验结果

|实验环境|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|---|
|苏州服务器|label,unlabel,test:260,3385,1563(tag频率<200,split:0.05,0.7)|Bert微调+多注意力+GAN（对抗指标：噪音样本与标签w近似度的和)|batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01;epoch_step:20|47.665|
