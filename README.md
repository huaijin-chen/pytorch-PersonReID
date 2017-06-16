# Triplet Margin Loss for Person Re-identification 

This Project is for Person Re-identification using [Triplet Loss](https://arxiv.org/abs/1503.03832) based on PyTorch

- [x] Triplet Margin Loss
- [x] load weights form darknet weights file
- [x] save weighes file with darknet format
- [x] find best threshold for test set
- [ ] more network structure 
- [ ] more trick using in ReID
- [ ] faster data loader
- [ ] faster in multi-GPU
- [ ] load and save caffemode

## Training and validation

1. creat triplet list file and put it in data/
2. set `image_root`=your/images/path

```
python train_val.py --gpus 0,1,2,3
```

## Reference

- FaceNet: A Unified Embedding for Face Recognition and Clustering


