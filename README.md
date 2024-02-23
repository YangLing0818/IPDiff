# IPDiff-ICLR 2024
Official implementation for our ICLR 2024 paper [Protein-Ligand Interaction Prior for Binding-aware 3D Molecule Diffusion Models](https://openreview.net/forum?id=qH9nrMNTIW).


![Alt text](image.png)

### Environment

```shell
conda env create -f ipdiff.yaml
conda activate ipdiff
```

### Data and Preparation
The data preparation follows [TargetDiff](https://arxiv.org/abs/2303.03543). For more details, please refer to [the repository of TargetDiff](https://github.com/guanjq/targetdiff?tab=readme-ov-file#data).

### Path to Pretrained IPNet:

```shell
./pretrained_models
```

### Training

```shell
conda activate ipdiff
python train.py
```

### Sampling

```shell
python sample_split.py --start_index 0 --end_index 99 --batch_size 25
```

### Evaluation

```shell
python eval_split.py --eval_start_index 0 --eval_end_index 99
```

### Calculate metrics

```shell
python cal_metrics_from_pt.py
```

## Citation
```
@inproceedings{huang2024proteinligand,
  title={Protein-Ligand Interaction Prior for Binding-aware 3D Molecule Diffusion Models},
  author={Zhilin Huang and Ling Yang and Xiangxin Zhou and Zhilong Zhang and Wentao Zhang and Xiawu Zheng and Jie Chen and Yu Wang and Bin CUI and Wenming Yang},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
