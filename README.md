## Context-Dependent Semantic Parsing over Temporally Structured Data

This repository contains the datasets and code associated with the paper:

Charles Chen, and Razvan Bunescu. **_Context-Dependent Semantic Parsing over Temporally Structured Data._**
In NAACL 2019 (Oral Presentation) [[PDF](https://arxiv.org/abs/1905.00245)]



# Citation

If you use our code in your research, please use the following BibTeX entry:

```
@inproceedings{chen2019context,
  title={Context-Dependent Semantic Parsing over Temporally Structured Data},
  author={Chen, Charles and Bunescu, Razvan},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
  pages={1--11},
  year={2019}
}
```


# Architecture

![alt text](https://github.com/charleschen1015/SemanticParsing/blob/master/SPAAC.png)



# PhysioGraph

![alt text](https://github.com/charleschen1015/SemanticParsing/blob/master/PhysioGraph.png)



# Requirements

* Python 2.7

* Tensorflow 0.9

* Numpy 1.14

* tqdm


# Usages

* The folders correspond to the models used in our paper: SeqGen, SeqGenAtt2In, SPAAC-MLE, and SPAAC-RL.


* The folder data contains both Real Interaction and Artificial Interaction Datasets.



## Training

python main.py --phase='train' 

OR

python main.py --phase='train' --load --model_file='pathtosavedmodel'



## Testing

python main.py --phase='test' --load --model_file='pathtosavedmodel'



# Contact

If you have any questions, please email me at lc971015@ohio.edu.



# GPU

All the experiments in our paper are performed with an NVIDIA GeForce GTX 1080 GPU. 
