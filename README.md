# Improved Algorithms for Neural Active Learning

In this repository, we provide the implementation of I-NeurAL.

## Prerequisites 

python 3.7, CUDA 11.2, torch 1.8.0, numpy 1.16.2

## Dataset

https://drive.google.com/file/d/1stur2RGGTEFfBGZipz4FAO_YHkRV0Zpw/view?usp=share_link

## Usage

rand.py: random baseline   
margin.py: margin baseline  
ntk-f.py: NeuAL-NTK-F (Algorithm 1 in [1])  
ntk-d.py: NeuAL-NTK-D (Algorithm 3 in [1])  
alps.py: ALPS Algorithm in [2]  
I-NeurAL.py: Our proposed I-NeurAL  

For example, to run I-NeurAL, use "python I-NeurAL.py"  

[1] Z. Wang, P. Awasthi, C. Dann, A. Sekhari, and C. Gentile. Neural active learning with
performance guarantees. Advances in Neural Information Processing Systems, 34, 2021.  
[2] G. DeSalvo, C. Gentile, and T. S. Thune. Online active learning with surrogate loss functions.
Advances in Neural Information Processing Systems, 34, 2021.  

#### Multi-Classification

In the experiments of our paper, we focus on the binary classification setting. In the "Multi" folder, we also provide the implementation of our method for multi-classification setting. The dataset link is https://drive.google.com/file/d/1dcbJ6q7KlZhckF2eyP4vCrM-TD1W8F4S/view?usp=sharing



