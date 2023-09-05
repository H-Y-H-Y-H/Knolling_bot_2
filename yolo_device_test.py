# Ultralytics YOLO 🚀, GPL-3.0 license

from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor


class PosePredictor(DetectionPredictor):

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes,
                                        nc=len(self.model.names))

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, shape)
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(
                Results(orig_img=orig_img,
                        path=img_path,
                        names=self.model.names,
                        boxes=pred[:, :6],
                        keypoints=pred_kpts))
        return results


def predict(cfg=DEFAULT_CFG, use_python=False):
    # model = cfg.model or 'yolov8n-pose.pt'
    model = "./models/830_pile_real_box/weights/best.pt"
    # source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
    #     else 'https://ultralytics.com/images/bus.jpg'
    source_pth = '../knolling_dataset/MLP_unstack_902/sim_images'
    args = dict(model=model, source=source_pth, save=True, save_txt=True, device='cuda:1')
    use_python = True
    if use_python:
        from ultralytics import YOLO
        zzz = YOLO(model)(**args)
        print(zzz)
    else:
        predictor = PosePredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict()
    # from ultralytics import YOLO
    #
    # model_path = "/home/ubuntu/Desktop/YOLOv8/runs/pose/train4/weights/"
    #
    # # Load a model
    # model = YOLO('yolov8n-knolling.yaml')  # build from YAML and transfer weights
    # model.load(model_path+'last.pt')
    #
    # source_pth = '/home/ubuntu/Desktop/datasets/knolling_data/images/val'
    #
    #
    # result = model(source=source_pth,conf = 0.5,save=True)
    # print(result)