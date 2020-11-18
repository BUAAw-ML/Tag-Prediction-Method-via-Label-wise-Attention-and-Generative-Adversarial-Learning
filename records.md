# Experimental progress and records

实验结果

|实验环境|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|---|
|苏州服务器|label,unlabel,test:260,3385,1563(tag频率<200,split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN（对抗指标：噪音样本与标签w近似度的和);未使用unlabel训练数据|epoch:20,epoch_step:13;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|38.292,37.583,38.411,38.042,37.163,35.323,34.915,37.344|
|614服务器(conda:discrete)|label,unlabel,test:260,3385,1563(tag频率<200,split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN（对抗指标：噪音样本与标签w近似度的和);D训练未使用G处理后的噪音数据)|epoch:20,epoch_step:13;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|47.369,46.496|
|---|---|---|---|---|
|苏州服务器|label,unlabel,test:260,3385,1563(tag频率<200,split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN（对抗指标：噪音样本与标签w近似度的和)|epoch:20,epoch_step:13;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|47.665,46.156,48.494(还在微涨),47.790(还在微涨)|
|苏州服务器|label,unlabel,test:260,3385,1563(tag频率<200,split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN（对抗指标：噪音样本与标签w近似度的和)|epoch:50,epoch_step:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|51.008,51.994|
|614服务器(conda:discrete)|label,unlabel,test:260,3385,1563(tag频率<200,split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN（对抗指标：噪音样本与标签w近似度的和)|epoch:20,epoch_step:13;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|50.9,47.756|
|614服务器(conda:discrete)|label,unlabel,test:260,3385,1563(tag频率<200,split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN（对抗指标：噪音样本与标签w近似度的和)|epoch:50,epoch_step:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|51.422,51.364|

