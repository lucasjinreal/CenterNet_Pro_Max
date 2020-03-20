# CenterNet Pro Max

> why this name? Because this repo based on centernet-better while there was somebody opensource another implementation which called centernet-better-plus, so that we have to using this name: centernet_pro_max. 

this repo is the reconstruct of original CenterNet. Unlike most implementation base on detectron2 or mmdetection, highly modulized code base makes users hard to understand what's the basic idea going on. So in this repo, we make it  as simply as possible, and let you can customized any idea or any new architecture you have.

This version build upon Centernet-Better, but unlike original repo, we provide something else:

- [x] Without any lib (not based on detectron2 or based on dl_lib), it's just single folder contains some modeling scripts;
- [x] Train is more intuitive, you can follow `train.py` and `centernet.py` to debug your own model architecture and experiments on your own losses or heads;
- [x] **We provide demo scripts to detect and visualize**;
- [x] We ported DCN from mmdetection with latest updates (this part is not like centernet-better);
- [x] We provide single GPU training settings (for some smaller datasets 1 GPU is enough, also you can using 8 GPUs as well);
- [ ] **We will provide onnx export**.
- [ ] We will provide onnx export and TensorRT inference;
- [ ] More backbones such as Vovnets;
- [ ] **More heads such as 3D and mask and CenterFace**;
- [x] CenterFace model ready;

**Please start and fork and watching this repo to subscribe latest updates!**



## Updates

- *2050.01.01*: more news to come;

- *2050.03.21*: Thanks for issue: [#3](https://github.com/jinfagang/CenterNet_Pro_Max/issues/3) pointed out, gaussian radius calculate method has been updated. What's gaussian radius? From my perspective, we want keep all alternative boxes that top left and right bottom point with some range, this picture can explain this:

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
python demo.py
```

![](https://s1.ax1x.com/2020/03/19/8rWijK.png)

![](https://s1.ax1x.com/2020/03/19/8rW8Hg.png)



![](https://s1.ax1x.com/2020/03/19/8rWa3q.png)

*note*: As you can see, **CenterNet** is very good at detect very small objects, I intended place these images here, if you try any other anchor based model such as yolov3 or retinanet even maskrcnn, it all will fail at such small objects! 




| **Backbone** | **Head**    | FPS(GTX1080ti) | mAP               | model link                                                   |
| ------------ | ----------- | -------------- | ----------------- | ------------------------------------------------------------ |
| resnet50     | without DCN |                | 35.7 (**3.1%** )↑ | [model](https://drive.google.com/open?id=1QJaMpT5WPC1XrrptOvoUSFLC1ww9k9qu) |
| resnet50     | with DCN    |                |                   |                                                              |
| resnet18     | without DCN |                |                   |                                                              |

*the model linked above maybe updates in the future, so pls subscribe our updates!



## Train

You can simply train the model by change your import way:

```
# default using config from configs.ct_coco_r50_config import config
python train.py
```





## Reference

thanks to original author of CenterNet-Better, and there also some implementations such as CenterNet-Bettter-Plus, but keep in mind that CenterNet-Pro-Max is always the best!