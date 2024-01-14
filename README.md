# Fourier Analysis on Robustness of Graph Convolutional Neural Networks for Skeleton-based Action Recognition

## Abstruct
Using Fourier analysis, we explore the robustness and vulnerability of graph convolutional neural networks (GCNs) for skeleton-based action recognition. We adopt a joint Fourier transform (JFT), a combination of the graph Fourier transform (GFT) and the discrete Fourier transform (DFT), to examine the robustness of adversarially-trained GCNs against adversarial attacks and common corruptions. Experimental results with the NTU RGB+D dataset reveal that adversarial training does not introduce a robustness trade-off between adversarial attacks and low-frequency perturbations, which typically occurs during image classification based on convolutional neural networks. This finding indicates that adversarial training is a practical approach to enhancing robustness against adversarial attacks and common corruptions in skeleton-based action recognition. Furthermore, we find that the Fourier approach cannot explain vulnerability against skeletal part occlusion corruption, which highlights its limitations. These findings extend our understanding of the robustness of GCNs, potentially guiding the development of more robust learning methods for skeleton-based action recognition.

## Description
This code is the source code of our paper "Fourier Analysis on Robustness of Graph Convolutional Neural Networks for Skeleton-based Action Recognition".

Our Paper is [here](https://arxiv.org/abs/2305.17939).

In this code, we use [NTU RGB+D dataset](https://arxiv.org/pdf/1604.02808.pdf) and [ST-GCN model](https://arxiv.org/abs/1801.07455).

## Notice
This code was custumized [the source code of ST-GCN](https://github.com/yysijie/st-gcn).  
In data preparation, We also used [the code of TCA-GCN](https://github.com/OrdinaryQin/TCA-GCN/tree/main).

Many thanks to the authors for all their works.

## Installation
You can download NTU RGB+D dataset from official website, and you place it in `./dataset/data/ntu/ntu_raw/`.
Then, run;
```
cd ./dataset/data/ntu
python get_raw_skes_data.py
python get_raw_denoised_data.py
python seq_transformation.py
cd ../../..
```


## Training
To train the model, run;
```
python main.py recognition -c config/st_gcn/ntu-xsub/train_joint.yaml
```
If you train the model by adversarial training, run;
```
python main.py recognition -c config/st_gcn/ntu-xsub/train_free_joint.yaml
```

## Testing
To test the models, run;
```
python main.py recognition -c config/st_gcn/ntu-xsub/test_joint.yaml
python main.py recognition -c config/st_gcn/ntu-xsub/test_free_joint.yaml
```
## Adversarial Attack
To attack the models by PGD, run;
```
python main.py recognition -c config/st_gcn/ntu-xsub/attack_joint.yaml
python main.py recognition -c config/st_gcn/ntu-xsub/attack_free_joint.yaml
```

## Common Corruptions
To make corrupted data, run;
```
cd ./dataset
python get_occluded.py
cd ..
```

## Visualization of frequency spectrum
To get frequency spectrum of input data, run;
```
cd ./dataset
python get_frequency.py
cd ..
```
To get the spectrum of corruption, run;
```
cd ./dataset
python get_crrup_fre.py
cd ..
```
To get the spectrum of adversarial perturbation, run;

```
cd ./dataset
python get_ad_fre.py
cd ..
```
