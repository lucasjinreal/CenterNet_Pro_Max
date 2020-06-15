# CenterNet Pro Max

updates.
**code deprecation!!! due some internal issue, this code is close-source, sorry for it!!**

For anybody wanna ask/consult/obtain any resource of CenterNet_Pro_Max I'd like to help u even though we close-source this version. Also welcome community join this dicuss platform to talk about AI:

http://t.manaai.cn

> why this name? Because this repo based on centernet-better while there was somebody opensource another implementation which called centernet-better-plus, so that we have to using this name: centernet_pro_max. 

this repo is the reconstruct of original CenterNet. Unlike most implementation base on detectron2 or mmdetection, highly modulized code base makes users hard to understand what's the basic idea going on. So in this repo, we make it  as simply as possible, and let you can customized any idea or any new architecture you have.

This version build upon Centernet-Better, but unlike original repo, we provide something else:

- [x] Without any lib (not based on detectron2 or based on dl_lib), it's just single folder contains some modeling scripts;
- [x] Train is more intuitive, you can follow `train.py` and `centernet.py` to debug your own model architecture and experiments on your own losses or heads;
- [x] **We provide demo scripts to detect and visualize**;
- [x] We ported DCN from mmdetection with latest updates (this part is not like centernet-better);
- [x] We provide single GPU training settings (for some smaller datasets 1 GPU is enough, also you can using 8 GPUs as well);
- [x] **We will provide onnx export**.
- [ ] We will provide onnx export and TensorRT inference;
- [ ] More backbones such as Vovnets;
- [ ] **More heads such as 3D and mask and CenterFace**;
- [x] CenterFace model ready;

**Please start and fork and watching this repo to subscribe latest updates!**



## Updates

- *2050.01.01*: more news to come;

- *2020.03.25*: Help wanted!

  Currently export onnx requires this function:

  ```python
  def gather_feature(fmap, index, mask=None, use_transform=False):
      if use_transform:
          # change a (N, C, H, W) tenor to (N, HxW, C) shape
          batch, channel = fmap.shape[:2]
          fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()
  
      dim = fmap.size(-1)
      index  = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
      fmap = fmap.gather(dim=1, index=index)
      if mask is not None:
          # this part is not called in Res18 dcn COCO
          mask = mask.unsqueeze(2).expand_as(fmap)
          fmap = fmap[mask]
          fmap = fmap.reshape(-1, dim)
      return fmap
  ```

  Which will involved OneHot op in onnx, is there way to avoid it? Anyone knows why probably can open an issue and PR are welcome!

- *2020.03.24*: Training on **any coco like dataset**:

  ```
  register_coco_instances('coco_tl', {}, './datasets/coco_tl/annotations/instances_train2017.json', './datasets/coco_tl/images')
  MetadataCatalog.get("coco_tl").thing_classes = categories
  ```

  Using 2 line of codes, you can train your custom dataset freely! More example see our `train_tl.py`.

- *2020.03.23*: We have supported ONNX export! Now this exported onnx is an experimental support since we merged all post process into onnx, there may be some unsupported op in other frameworks. here is current onnx model ops:

  ```
  > onnxexp centernet_r50_coco.onnx summary
  Exploring on onnx model: centernet_r50_coco.onnx
  ONNX model sum on: centernet_r50_coco.onnx
  
  
  -------------------------------------------
  ir version: 6
  opset_import: 9 
  producer_name: pytorch
  doc_string: 
  all ops used: Constant,Gather,Shape,Cast,Or,Add,Unsqueeze,Concat,Reshape,ConstantOfShape,Mul,Equal,Where,Expand,NonZero,Transpose,Squeeze,Slice,ATen,Conv,BatchNormalization,Relu,MaxPool,ConvTranspose,Sigmoid,TopK,Flatten,Div
  -------------------------------------------
  ```

  Be note that, Nonzero not supported by onnx2trt, we will polish onnx model make it simply enough to deploy!

  Also, we have support a custom dataset training which is nuScenes! Checkout our codes to try yourself!

- *2020.03.21*: Thanks for issue: [#3](https://github.com/jinfagang/CenterNet_Pro_Max/issues/3) pointed out, gaussian radius calculate method has been updated. What's gaussian radius? From my perspective, we want keep all alternative boxes that top left and right bottom point with some range, this picture can explain this:

  ![](https://pic3.zhimg.com/80/v2-2c6dcd69318e8650eddab6a4c82407ba_720w.jpg)

  we want keep all boxes that corner point within a certain range, how to calculate this range? We using gaussian radius, in code we updated to:

  ```python
  @staticmethod
  def get_gaussian_radius(box_size, min_overlap):
      """
      copyed from CornerNet
      box_size (w, h), it could be a torch.Tensor, numpy.ndarray, list or tuple
      notice: we are using a bug-version, please refer to fix bug version in CornerNet
      """
      box_tensor = torch.Tensor(box_size)
      width, height = box_tensor[..., 0], box_tensor[..., 1]
  
      a1  = 1
      b1  = (height + width)
      c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
      sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
      # r1  = (b1 + sq1) / 2
      r1 = (b1 - sq1)/(2*a1)
  
      a2  = 4
      b2  = 2 * (height + width)
      c2  = (1 - min_overlap) * width * height
      sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
      # r2  = (b2 + sq2) / 2
      r1 = (b2 - sq2) / (2*a2)
  
      a3  = 4 * min_overlap
      b3  = -2 * min_overlap * (height + width)
      c3  = (min_overlap - 1) * width * height
      sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
      # r3  = (b3 + sq3) / 2
      r3 = (b3 + sq3) / (2*a3)
      return torch.min(r1, torch.min(r2, r3))
  ```

  more info can refer to that issue.

- *2020.03.20*: CenterFace model supported!.

- *2020.03.19*: First release the codes, meanwhile centerface model architecture has been added in.



## Demo

We have provide resnet50 pretrained weights and resnet101 pretrained weights (head without DCN), to run demo visualize, simply:

```
sudo pip3 install alfred-py
python demo.py
```

`alfred-py` is a deep learning util lib for visualization and common utils, github url: https://github.com/jinfagang/alfred, you can easy install from pip.

![](https://s1.ax1x.com/2020/03/19/8rWijK.png)

![](https://s1.ax1x.com/2020/03/19/8rW8Hg.png)



![](https://s1.ax1x.com/2020/03/19/8rWa3q.png)

*note*: As you can see, **CenterNet** is very good at detect very small objects, I intended place these images here, if you try any other anchor based model such as yolov3 or retinanet even maskrcnn, it all will fail at such small objects! 




| **Backbone** | **Head**    | FPS(GTX1080ti) | mAP               | model link                                                   |
| ------------ | ----------- | -------------- | ----------------- | ------------------------------------------------------------ |
| resnet50     | without DCN |                | 35.7 (**3.1%** )↑ | [model](https://drive.google.com/open?id=1QJaMpT5WPC1XrrptOvoUSFLC1ww9k9qu) [newest model](https://share.weiyun.com/5w85xMm) |
| resnet50     | with DCN    |                |                   |                                                              |
| resnet18     | without DCN |                |                   |                                                              |
| resnet18     | with DCN    |                | -                 |                                                              |
| volvenet39   |             |                |                   |                                                              |
| mobilenetv3  |             |                |                   |                                                              |

the model linked above maybe updates in the future, so pls subscribe our updates! CenterFace will update once we finished training. 



## Train

You can simply train the model by change your import way:

```
# default using config from configs.ct_coco_r50_config import config
python train.py
```





## Reference

thanks to original author of CenterNet-Better, and there also some implementations such as CenterNet-Bettter-Plus, but keep in mind that CenterNet-Pro-Max is always the best!
