

#实验结果 Experimental progress and records

##programmerWeb数据集

###tag频率<200的数据集

|实验环境|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|---|
|614服务器(conda:discrete)|label,unlabel,test:3645,0,1563(split:0.7,未加title_ids)|Bert微调+多注意力|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.01,B0.1|64.618|
|苏州服务器|label,unlabel,test:3645,0,1563(split:0.7,未加title_ids)|Bert微调+多注意力|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.01,B0.1|65.03|
|614服务器(conda:discrete)|label,unlabel,test:2604,0,1563(split:0.5,未加title_ids)|Bert微调+多注意力;未使用unlabel训练数据|epoch:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|62.620|
|614服务器(conda:discrete)|label,unlabel,test:1562,0,1563(split:0.3,未加title_ids)|Bert微调+多注意力;未使用unlabel训练数据|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|58.845|
|苏州服务器|label,unlabel,test:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力;未使用unlabel训练数据|epoch:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|38.292  37.583  38.411  38.042  37.163  35.323  34.915  37.344|
|苏州服务器|label,unlabel,test:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力;未使用unlabel训练数据|epoch:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|40.888  39.399  37.532|
|---|---|---|---|---|
|614服务器(conda:discrete)|label,unlabel,test:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN（对抗指标：噪音样本与标签w近似度的和);D训练未使用G处理后的噪音数据|epoch:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|47.369  46.496|
|614服务器(conda:discrete)|label,unlabel,test:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN（对抗指标：噪音样本与标签w近似度的和);D训练未使用G处理后的噪音数据|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|47.315  47.826  47.296|
|苏州服务器|label,unlabel,test:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN(对抗指标：噪音样本与标签w近似度的和)|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|51.008  51.994  49.915  51.133|
|苏州服务器|label,unlabel,test:520,3125,1563(tsplit:0.1,0.7,未加title_ids)|Bert微调+多注意力+GAN(对抗指标：噪音样本与标签w近似度的和)|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|54.454|
|苏州服务器|label,unlabel,test:1040,2605,1563(split:0.2,0.7,未加title_ids)|Bert微调+多注意力+GAN(对抗指标：噪音样本与标签w近似度的和)|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|58.86|
|614服务器(conda:discrete)|label,unlabel,test:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN(对抗指标：噪音样本与标签w近似度的和)|epoch:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|50.9  47.756|
|614服务器(conda:discrete)|label,unlabel,test:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN(对抗指标：噪音样本与标签w近似度的和)|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|51.422  51.364 51.543|
|苏州服务器|label,unlabel,test:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN(对抗指标：噪音样本与标签w近似度的和)；Generator设置2层（没提的都为1层）|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|52.123|


另外，进行了其它试验，包括：
- 模型在Gnerator学习率为0.0001，0.01下效果不好;
- Gnerator用了G_feat_match效果不好；
- Gnerator设置3层不好，使用dropout不好；

###全部数据集

|实验环境|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|---|
|苏州服务器|label,unlabel,test:612,0,1226(split:0.05,0.9,未加title_ids)|Bert微调+多注意力|epoch:50;epoch_step:13;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|40.378|
|苏州服务器|label,unlabel,test:8579,0,1226(split:0.7,0.9,未加title_ids)|Bert微调+多注意力|epoch:50;epoch_step:13;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|59.774|
|苏州服务器|label,unlabel,test:11030,0,1226(split:0.7,0.9,未加title_ids)|Bert微调+多注意力|epoch:50;epoch_step:13;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|62|
|苏州服务器|label,unlabel,test:612,,1226(split:0.05,0.9,未加title_ids)|Bert微调+多注意力+GAN(对抗指标：噪音样本与标签w近似度的和)|epoch:50;epoch_step:45；batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|44|
|苏州服务器|label,unlabel,test:2448,,1226(split:0.05,0.9,未加title_ids)|Bert微调+多注意力+GAN(对抗指标：噪音样本与标签w近似度的和)|epoch:50;epoch_step:45；batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|51.098|

小结：
- 采用全部数据集（115个标签）时，提出的方法的效果只好大概百分之四，不是很明显；

###tag频率<100的数据集

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|label,unlabel,test:72,0,438(split:0.05,0.7,未加title_ids)|Bert微调+多注意力|epoch:20;epoch_step:13;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|46.170  45.024  48.917  46.872  43.485  47.755|
|label,unlabel,test:72,949,438(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN(对抗指标：噪音样本与标签w近似度的和)|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|54.745  54.383  55.709  52.864|

另外，进行了其它试验，包括：
- model里的判别特征如何改成和权重矩阵乘后求mean()效果是不好的。
- 尝试了对所有未标注样本打伪标签（预测概率最大的类别tag设置为1），然后一起训练模型。但是基本训练不起来。

##gan-bert数据集

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|label,unlabel,test:109,0,500|Bert微调+多注意力|epoch:20;epoch_step:13;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|20  22  23|
|label,unlabel,test:109,5343,500|Bert微调+多注意力+GAN(对抗指标：噪音样本与标签w近似度的和)|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|28|


##AAPD数据集

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|label,unlabel,test:2742,,16452|Bert微调+多注意力|epoch:15;epoch_step:;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|44|
|label,unlabel,test:2742,35646,164520|Bert微调+多注意力+GAN(对抗指标：噪音样本与标签w近似度的和)|epoch:50;epoch_step:45;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|---|---|---|---|
|label,unlabel,test:548,,16452|Bert微调+多注意力|epoch:21;epoch_step:15;batch-size:30;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|28.793|
|label,unlabel,test:548,37840,16452|Bert微调+多注意力+GAN(对抗指标：噪音样本与标签w近似度的和)|epoch:10;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|34.139|
|label,unlabel,test:548,37840,16452|Bert微调+多注意力+GAN(对抗指标：噪音样本与标签w近似度的和)|epoch:15;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|31.651|
|---|---|---|---|
|label,unlabel,test:548,,3291|Bert微调+多注意力|epoch:50;epoch_step:15;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|34.768|
|label,unlabel,test:548,7129,3291|Bert微调+多注意力+GAN(对抗指标：噪音样本与标签w近似度的和)|epoch:23;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|39.414|
|---|---|---|---|
|label,unlabel,test:548,,3291|Bert微调+多注意力|epoch:50;epoch_step:15;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|34.129  32.716|
|label,unlabel,test:548,7129,3291|Bert微调+多注意力+GAN(对抗指标：噪音样本与标签w近似度的和)|epoch:23;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|40.419|
|---|---|---|---|
|label,unlabel,test:548,,3291|Bert微调+多注意力|epoch:50;epoch_step:15;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||

另外，进行了其它试验，包括：
- 0.69的label，0.01的。
- batch-size使用30时，GAN初期提不起来（6轮都不咋提高），感觉之后效果应该不好。
- 提出方法当模型达到最高性能后性能又会快速下降（好像是在d_loss变为0的时候）
- 感觉batch-size对方法的效果有影响