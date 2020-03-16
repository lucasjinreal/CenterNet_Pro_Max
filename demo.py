from configs.ct_coco_r50_config import config
from models.data import MetadataCatalog
from models.centernet import build_model
from models.train.checkpoint import DetectionCheckpointer
from models.data import transforms as T
import torch
from PIL import Image
import cv2
import sys
import os
from alfred.vis.image.det import visualize_det_cv2_part
from alfred.vis.image.get_dataset_label_map import coco_label_map_list
import glob


class DefaultPredictor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        print('try load weights from: {}'.format(cfg.MODEL.WEIGHTS))
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    @torch.no_grad()
    def __call__(self, original_image):
        # Apply pre-processing to image.
        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        # image = self.transform_gen.get_transform(original_image).apply_image(original_image)
        image = original_image
        # we will do preprocess inside model
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        predictions = self.model([inputs])[0]
        return predictions


if __name__ == '__main__':
    config.MODEL.WEIGHTS = 'weights/centernet_r50_coco.pth'
    # config.MODEL.WEIGHTS = 'checkpoints/resnet50_centernet.pth'
    predictor = DefaultPredictor(config)
    coco_label_map_list = coco_label_map_list[1:]

    if len(sys.argv) > 1:
        data_f = sys.argv[1]
    else:
        data_f = './images'
    if os.path.isdir(data_f):
        img_files = glob.glob(os.path.join(data_f, '*.jpg'))
        for img_f in img_files:
            ori_img = cv2.imread(img_f)
            b = predictor(ori_img)['instances']
            boxes = b.pred_boxes.tensor.cpu().numpy()
            scores = b.scores.cpu().numpy()
            classes = b.pred_classes.cpu().numpy()
            print('b.pred_boxes: {}'.format(boxes))
            print('b.scores: {}'.format(scores))
            print('b.pred_classes: {}'.format(classes))
            visualize_det_cv2_part(ori_img, scores, classes, boxes, class_names=coco_label_map_list, thresh=0.16,
                                   is_show=True)
    else:
        ori_img = cv2.imread(data_f)
        b = predictor(ori_img)['instances']
        boxes = b.pred_boxes.tensor.cpu().numpy()
        scores = b.scores.cpu().numpy()
        classes = b.pred_classes.cpu().numpy()
        print('b.pred_boxes: {}'.format(boxes))
        print('b.scores: {}'.format(scores))
        print('b.pred_classes: {}'.format(classes))
        visualize_det_cv2_part(ori_img, scores, classes, boxes, class_names=coco_label_map_list, thresh=0.16,
                               is_show=True)
