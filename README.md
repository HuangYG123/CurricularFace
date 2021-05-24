## CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition
Yuge Huang, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li, Feiyue Huang

This repository is the official PyTorch implementation of paper [CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition](https://arxiv.org/abs/2004.00288). (The work has been accepted by [CVPR2020](http://cvpr2020.thecvf.com/)).

We have released a training framework for face recognition, please refer to the details at [TFace](https://github.com/Tencent/TFace/).

## Main requirements

  * **torch == 1.1.0**
  * **torchvision == 0.3.0**
  * **tensorboardX == 1.7**
  * **bcolz == 1.2.1**
  * **Python 3**
  
## Usage
```bash
# To train the model:
sh train.sh
# To evaluate the model:
(1)please first download the val data in https://github.com/ZhaoJ9014/face.evoLVe.PyTorch.
(2)set the checkpoint dir in config.py
sh evaluate.sh
```
You can change the experimental setting by simply modifying the parameter in the config.py

## Model
The IR101 pretrained model can be downloaded here. 
[Baidu Cloud](link: https://pan.baidu.com/s/1bu-uocgSyFHf5pOPShhTyA 
passwd: 5qa0), 
[Google Drive](https://drive.google.com/open?id=1upOyrPzZ5OI3p6WkA5D5JFYCeiZuaPcp)

## Result
The results of the released pretrained model are as follows:
|Data| LFW | CFP-FP | CPLFW | AGEDB | CALFW | IJBB (TPR@FAR=1e-4) | IJBC (TPR@FAR=1e-4) |
|:---:|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Result | 99.80 | 98.36 | 93.13 | 98.37 | 96.05 | 94.86 | 96.15 ||

The results are slightly different from the results in the paper because we replaced DataParallel with DistributedDataParallel and retrained the model.

## Citing this repository
If you find this code useful in your research, please consider citing us:
```
@article{huang2020curricularface,
	title={CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition},
	author={Yuge Huang and Yuhan Wang and Ying Tai and  Xiaoming Liu and Pengcheng Shen and Shaoxin Li and Jilin Li, Feiyue Huang},
	booktitle={CVPR},
	pages={1--8},
	year={2020}
}
```

## Contacts
If you have any questions about our work, please do not hesitate to contact us by emails.
Yuge Huang: yugehuang@tencent.com
Ying Tai: yingtai@tencent.com



