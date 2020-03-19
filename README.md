# CenterNet Pro Max

> why this name? Because this repo based on centernet-better while there was somebody opensource another implementation which called centernet-better-plus, so that we have to using this name: centernet_pro_max. 

this repo is reconstruct of original center. Unlike most implementation base on detectron2 or mmdetection, highly modulized code base make users hard to understand what's the basic idea goes. So this repo is main make it simply as much as possible, and make you can customized any idea or any new architecture you have.

This version build upon Centernet-Better, but unlike this repo, we provide something else:

- [x] Without any lib (not based on detectron2 or based on dl_lib), it's just single folder contains some modeling scripts;
- [x] Train is more intuitive, you can follow `train.py` and `centernet.py` to debug your own model architecture and experiments on your own losses or heads;
- [x] We provide demo scripts to detect and visualize;
- [x] We ported DCN from mmdetection with latest updates (this part is not like centernet-better);
- [x] We provide single GPU training settings (for some smaller datasets 1 GPU is enough, also you can using 8 GPUs as well);
- [ ] We provide onnx export.
- [ ] We will provide onnx export and TensorRT inference;
- [ ] More backbones such as Vovnets;
- [ ] More heads such as 3D and mask;
- [x] CenterFace model ready;



## Updates

- *2020.03.21*: more news to come.
- *2020.03.19*: First release the codes, meanwhile centerface model architecture has been added in.



## Demo

We have provide resnet50 pretrained weights and resnet101 pretrained weights (head without DCN), to run demo visualize, simply:

```
python demo.py
```

| **Backbone** | **Head**    | FPS(GTX1080ti) | mAP  | model link                                                   |
| ------------ | ----------- | -------------- | ---- | ------------------------------------------------------------ |
| resnet50     | without DCN |                |      | [model](https://drive.google.com/open?id=1QJaMpT5WPC1XrrptOvoUSFLC1ww9k9qu) |
| resnet50     | with DCN    |                |      |                                                              |
| resnet18     | without DCN |                |      |                                                              |



## Train

You can simply train the model by change your import way:

```
# default using config from configs.ct_coco_r50_config import config
python train.py
```





## Reference

thanks to original author of CenterNet-Better, and there also some implementations such as CenterNet-Bettter-Plus, but keep in mind that CenterNet-Pro-Max is always the best!