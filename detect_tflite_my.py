import numpy as np
import cv2
from PIL import Image
from time import time
import torch
from ultralytics.utils import ops
from ultralytics.engine.results import Results
from tflite_runtime.interpreter import Interpreter, load_delegate



class Sushi_Plate_Detector:
    def __init__(self, path, conf=0.6, iou=0.35) -> None:
        self.path = path
        self.model = None
        self.input_id = None
        self.output_id = None
        self.wh = None
        self.classes = ['100 NTD', '30 NTD', '40 NTD', '60 NTD', '80 NTD', 'food']
        self.agnostic_nms = True
        self.max_det = 300
        self.consider_classes = [0, 1, 2, 3, 4]
        self.conf = conf
        self.iou = iou
        self._load_model()
        self._warmup()

    def detect(self, img):
        x = self._preprocess(img)
        self.model.set_tensor(self.input_id, x)
        self.model.invoke()
        y = self.model.get_tensor(self.output_id)
        plotted, boxes, cls_ids = self._postprocess(y, img)
        return plotted, boxes, cls_ids

    def _preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        dst_w, dst_h = self.wh
        h, w = img.shape[:2]
        scale_w, scale_h = dst_w / w, dst_h / h
        scale = min(scale_w, scale_h)
        new_h, new_w = int(h * scale), int(w * scale)

        if scale_w < scale_h:
            dx, dy = [0, (dst_h - new_h) // 2]  # x, y
        else:
            dx, dy = [(dst_w - new_w) // 2, 0]  # x, y
        
        img = cv2.resize(img, (new_w, new_h))
        x = np.full((dst_h, dst_w, 3), 114, 'uint8')
        x[dy : dy + new_h, dx : dx + new_w] = img
        x = x / 255.
        x = x.astype('float32')[None]
        return x

    def _postprocess(self, preds, img):
        """Post-processes predictions and returns a list of Results objects."""
        if not isinstance(preds, torch.Tensor):
            preds = torch.from_numpy(preds)
        
        preds = ops.non_max_suppression(
            preds,
            self.conf,
            self.iou,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
            classes=self.consider_classes
        )[0]
        preds[:, :4] *= torch.tensor([*self.wh, *self.wh])
        preds[:, :4] = ops.scale_boxes(self.wh[::-1], preds[:, :4], img.shape[:2])
        plotted, boxes, cls_ids = self._plot(preds, img)
        return plotted, boxes, cls_ids

    def _load_model(self):
        #self.model = Interpreter(self.path, experimental_delegates=[load_delegate('/usr/lib/gpu_external_delegate.so')])
        ext_delegate_options = {}
        options = 'backends:GpuAcc,CpuAcc'.split(';')
        for o in options:
            kv = o.split(':')
            if (len(kv) == 2):
                ext_delegate_options[kv[0].strip()] = kv[1].strip()
            else:
                raise RuntimeError('Error parsing delegate option: ' + o)
        #self.model = Interpreter(self.path, experimental_delegates=[load_delegate('/usr/lib/gpu_external_delegate.so')])
        self.model = Interpreter(self.path, experimental_delegates=[load_delegate('/usr/lib/libarmnnDelegate.so.29', ext_delegate_options)])
        self.input_id = self.model.get_input_details()[0]['index']
        self.output_id = self.model.get_output_details()[0]['index']
        self.wh = self.model.get_input_details()[0]['shape'][1:3][::-1]
        self.model.allocate_tensors()

    def _plot(self, boxes, origin_img):
        result = Results(origin_img, path='', names=self.classes, boxes=boxes)
        plotted = result.plot(conf=True,
                              labels=True,
                              probs=True,
                              save=False)
        boxes = result.boxes.xyxy
        scores = result.boxes.conf
        cls_ids = result.boxes.cls.to(torch.int32)
        boxes = np.c_[boxes, scores]
        return plotted, boxes, cls_ids

    def _warmup(self):
        x = np.random.randint(0, 255, (*self.wh[::-1], 3), 'uint8')
        t0 = time()
        self.detect(x)
        t1 = time()
        print('warmup time:', t1 - t0)


def main():
    IMAGE_PATH = '2e4c251f-2024-11-20_993.jpg'
    MODEL_PATH = 'weight/best_float32.tflite'

    model = Sushi_Plate_Detector(MODEL_PATH)

    img = cv2.imread(IMAGE_PATH)
    plotted = model.detect(img)
    cv2.imshow('aaa', plotted)
    cv2.waitKey(0)



if __name__ == '__main__':
    main()