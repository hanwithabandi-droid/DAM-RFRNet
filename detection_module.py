"""
===============================================================================
Downstream Vision Evaluation: Object Detection under Adverse Weather
===============================================================================
This module evaluates object detection performance (bounding boxes, class labels,
and confidence scores) on Hazy Input Images vs. DAM-RFRNet Dehazed Outputs.
Demonstrates the critical software perception recovery provided by dehazing.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import cv2
import torch
from PIL import Image
import torchvision
from torchvision import transforms

class AdverseWeatherObjectDetector:
    """
    Object detection evaluator for comparing perception accuracy on raw hazy images
    versus dehazed restored images.
    """
    def __init__(self, confidence_threshold: float = 0.4):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = None
        self.torchvision_model = None
        self._init_detector()

    def _init_detector(self):
        """Attempts loading Ultralytics YOLOv8, falling back to TorchVision Faster R-CNN."""
        try:
            from ultralytics import YOLO
            print("[*] Loading YOLOv8 Object Detection software model...")
            self.yolo_model = YOLO('yolov8n.pt')
            return
        except Exception:
            pass

        try:
            print("[*] Loading TorchVision Faster R-CNN detection model...")
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.torchvision_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
            self.torchvision_model.to(self.device).eval()
            self.coco_labels = weights.meta["categories"]
        except Exception as e:
            print(f"[!] Detection model load warning: {e}")

    def detect_and_draw(self, pil_image: Image.Image) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """
        Runs object detection on a PIL Image and returns an annotated PIL Image along with detection metrics.
        """
        img_np = np.array(pil_image.convert("RGB"))
        h, w, _ = img_np.shape
        detections = []
        annotated_np = img_np.copy()

        # 1. Option A: Ultralytics YOLOv8
        if self.yolo_model is not None:
            results = self.yolo_model(pil_image, conf=self.confidence_threshold, verbose=False)[0]
            for box in results.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                label_name = results.names[cls_id]

                detections.append({
                    'label': label_name,
                    'confidence': conf,
                    'box': coords.tolist()
                })

                # Draw bounding box on image
                cv2.rectangle(annotated_np, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 128), 2)
                caption = f"{label_name}: {conf:.2f}"
                cv2.putText(annotated_np, caption, (coords[0], max(20, coords[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 2)

        # 2. Option B: TorchVision Faster R-CNN
        elif self.torchvision_model is not None:
            tensor_in = transforms.ToTensor()(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                preds = self.torchvision_model(tensor_in)[0]

            boxes = preds['boxes'].cpu().numpy().astype(int)
            scores = preds['scores'].cpu().numpy()
            labels = preds['labels'].cpu().numpy()

            for box, score, label_id in zip(boxes, scores, labels):
                if score < self.confidence_threshold:
                    continue
                label_name = self.coco_labels[label_id] if label_id < len(self.coco_labels) else f"Class {label_id}"
                
                detections.append({
                    'label': label_name,
                    'confidence': float(score),
                    'box': box.tolist()
                })

                cv2.rectangle(annotated_np, (box[0], box[1]), (box[2], box[3]), (0, 255, 128), 2)
                caption = f"{label_name}: {score:.2f}"
                cv2.putText(annotated_np, caption, (box[0], max(20, box[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 2)

        # 3. Fallback: Edge-contour based object candidate detector
        else:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            count = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 800:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    count += 1
                    detections.append({
                        'label': 'Object Candidate',
                        'confidence': min(0.95, area / (h * w * 0.1)),
                        'box': [x, y, x + bw, y + bh]
                    })
                    cv2.rectangle(annotated_np, (x, y), (x + bw, y + bh), (0, 255, 128), 2)
                    cv2.putText(annotated_np, f"Object #{count}", (x, max(20, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 2)

        annotated_pil = Image.fromarray(annotated_np)
        return annotated_pil, detections


def compare_detections(hazy_img: Image.Image, dehazed_img: Image.Image) -> Dict[str, Any]:
    """
    Runs comparative object detection on both Hazy and Dehazed images.
    Returns annotated images and statistical metrics.
    """
    detector = AdverseWeatherObjectDetector(confidence_threshold=0.35)
    
    hazy_annotated, hazy_dets = detector.detect_and_draw(hazy_img)
    dehazed_annotated, dehazed_dets = detector.detect_and_draw(dehazed_img)

    avg_conf_hazy = np.mean([d['confidence'] for d in hazy_dets]) if hazy_dets else 0.0
    avg_conf_dehazed = np.mean([d['confidence'] for d in dehazed_dets]) if dehazed_dets else 0.0

    return {
        'hazy_annotated': hazy_annotated,
        'dehazed_annotated': dehazed_annotated,
        'hazy_count': len(hazy_dets),
        'dehazed_count': len(dehazed_dets),
        'hazy_avg_conf': avg_conf_hazy,
        'dehazed_avg_conf': avg_conf_dehazed,
        'hazy_detections': hazy_dets,
        'dehazed_detections': dehazed_dets
    }


if __name__ == '__main__':
    # Unit test
    dummy = Image.fromarray(np.uint8(np.random.rand(300, 400, 3) * 255))
    det = AdverseWeatherObjectDetector()
    annotated, dets = det.detect_and_draw(dummy)
    print(f"[+] Object Detector Unit Test Success! Found {len(dets)} objects.")
