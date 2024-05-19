# CofiPara
Official PyTorch implementation for the paper - **CofiPara: A Coarse-to-fine Paradigm for Multimodal Sarcasm Target Identification with Large Multimodal Models**.

(**ACL 2024**: *The 62nd Annual Meeting of the Association for Computational Linguistics, August 2024, Bangkok*.) [[`paper`](https://arxiv.org/pdf/2405.00390)]

## Environment
```
python==3.10
torch==2.0.1
(pip install -r requirements.txt)
```
Please also install groudingdino in [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) repository.

## Dataset

Please refer to [MSD_Data](https://github.com/Lbotirx/CofiPara/tree/master/data/mmsd2/README.md) and [MSTI_Data](https://github.com/Lbotirx/CofiPara/tree/master/data/msti/README.md)

## Training and Test
Edit model_config.py to change training and test settings.
```
python main.py
```

## Citation

```
@inproceedings{cofipara2024msti,
  title={CofiPara: A Coarse-to-fine Paradigm for Multimodal Sarcasm Target Identification with Large Multimodal Models},
  author={Lin, Hongzhan and Chen, Zixin and
    Luo, Ziyang and
    Cheng, Mingfei and
    Ma, Jing and
    Chen, Guang},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year={2024}
}
```
