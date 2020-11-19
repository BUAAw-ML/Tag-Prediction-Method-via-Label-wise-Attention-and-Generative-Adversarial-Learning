# Experimental progress and records

实验结果

|实验环境|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|---|
|614服务器(conda:discrete)|label,unlabel,test:3645,0,1563(tag频率<200,split:0.7,未加title_ids)|Bert微调+多注意力|epoch:50,epoch_step:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.01,B0.1|64.618|
|苏州服务器|label,unlabel,test:3645,0,1563(tag频率<200,split:0.7,未加title_ids)|Bert微调+多注意力|epoch:50,epoch_step:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.01,B0.1|65.03|
|614服务器(conda:discrete)|label,unlabel,test:2604,0,1563(tag频率<200,split:0.5,未加title_ids)|Bert微调+多注意力;未使用unlabel训练数据|epoch:40,epoch_step:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|62.620|
|614服务器(conda:discrete)|label,unlabel,test:1562,0,1563(tag频率<200,split:0.3,未加title_ids)|Bert微调+多注意力;未使用unlabel训练数据|epoch:50,epoch_step:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|58.845|
|苏州服务器|label,unlabel,test:260,3385,1563(tag频率<200,split:0.05,0.7,未加title_ids)|Bert微调+多注意力;未使用unlabel训练数据|epoch:20,epoch_step:13;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|38.292  37.583  38.411  38.042  37.163  35.323  34.915  37.344|
|苏州服务器|label,unlabel,test:260,3385,1563(tag频率<200,split:0.05,0.7,未加title_ids)|Bert微调+多注意力;未使用unlabel训练数据|epoch:20,epoch_step:13;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|40.888  39.399  37.532|
|614服务器(conda:discrete)|label,unlabel,test:260,3385,1563(tag频率<200,split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN（对抗指标：噪音样本与标签w近似度的和);D训练未使用G处理后的噪音数据|epoch:20,epoch_step:13;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|47.369  46.496|
|614服务器(conda:discrete)|label,unlabel,test:260,3385,1563(tag频率<200,split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN（对抗指标：噪音样本与标签w近似度的和);D训练未使用G处理后的噪音数据|epoch:50,epoch_step:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|47.315  47.826  47.296|
|---|---|---|---|---|
|苏州服务器|label,unlabel,test:260,3385,1563(tag频率<200,split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN(对抗指标：噪音样本与标签w近似度的和)|epoch:50,epoch_step:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|51.008  51.994  49.915  51.133|
|苏州服务器|label,unlabel,test:520,3125,1563(tag频率<200,split:0.1,0.7,未加title_ids)|Bert微调+多注意力+GAN(对抗指标：噪音样本与标签w近似度的和)|epoch:50,epoch_step:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|54.454|
|苏州服务器|label,unlabel,test:520,3125,1563(tag频率<200,split:0.1,0.7,未加title_ids)|Bert微调+多注意力+GAN(对抗指标：噪音样本与标签w近似度的和)|epoch:50,epoch_step:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|614服务器(conda:discrete)|label,unlabel,test:260,3385,1563(tag频率<200,split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN(对抗指标：噪音样本与标签w近似度的和)|epoch:20,epoch_step:13;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|50.9  47.756|
|614服务器(conda:discrete)|label,unlabel,test:260,3385,1563(tag频率<200,split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN(对抗指标：噪音样本与标签w近似度的和)|epoch:50,epoch_step:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|51.422  51.364 51.543|
|苏州服务器|label,unlabel,test:260,3385,1563(tag频率<200,split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN(对抗指标：噪音样本与标签w近似度的和)；Generator设置2层（没提的都为1层）|epoch:50,epoch_step:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|52.123|


另外，进行了其它试验，包括：
- 模型在Gnerator学习率为0.0001，0.01下效果不好;
- Gnerator用了G_feat_match效果不好；
- Gnerator设置3层不好，使用dropout不好；