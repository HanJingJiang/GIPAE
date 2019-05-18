GIPAE
===
Copyright (C) 2019 Han-Jing Jiang(jianghanjing17@mails.ucas.ac.cn),Yu-An Huang(yu-an.huang@connect.polyu.hk),Zhu-Hong You(zhuhongyou@xjb.ac.cn)

Computational drug repositioning
---
We here propose a drug repositioning computational method combining the techniques of Gaussian interaction profile kernel and Auto-encoder (GIPAE) which is able to learn new features effectively representing drug-disease associations via its hidden layers.In order to further reduce the computation cost, both batch normalization layer and the full-connected layer are introduced to reduce training complexity.

Dataset
---
1.CdiseaseSimilarity store disease similarity matrix of Cdataset
2.diseaseSimilarity store disease similarity matrix of Fdataset
3.Drug -disease-whole and c-drug-disease-whole store known drug-disease associations of Cdataset and Fataset.

code
---
1.Feature.py:Function to generate the total characteristics
2.NN.py:The features are obtained by the batch normalization layer and the full-connected layer 
3.RF.py:predict potential indications for drugs
All files of Dataset and Code should be stored in the same folder to run GIPAE.
