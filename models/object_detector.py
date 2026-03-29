"""Object Detector for identifying objects in video frames."""

import torch
import torch.nn as nn
import torchvision.models.detection as det_models
import numpy as np

COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


class ObjectDetector:
    """Wraps a pretrained Faster R-CNN for object detection in video frames."""

    def __init__(self, confidence_threshold: float = 0.5, device: str = "cpu"):
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.model = det_models.fasterrcnn_resnet50_fpn(
            weights=det_models.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        self.model.to(self.device)
        self.model.eval()
        self.class_names = COCO_CLASSES

    @torch.no_grad()
    def detect_frame(self, frame_tensor: torch.Tensor) -> list[dict]:
        """Detect objects in a single frame. Input: (C, H, W) float [0,1]. Returns list of detections."""
        outputs = self.model([frame_tensor.to(self.device)])[0]
        detections = []
        for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
            if score >= self.confidence_threshold:
                detections.append({
                    "box": box.cpu().numpy(),
                    "label": self.class_names[label.item()],
                    "label_id": label.item(),
                    "score": score.item(),
                })
        return detections

    def detect_video(self, frames: torch.Tensor) -> dict:
        """Detect objects across all frames of a video.

        Args:
            frames: (T, C, H, W) float tensor [0,1]

        Returns:
            dict with:
                - 'per_frame': list of per-frame detection lists
                - 'summary': {class_name: {'count': int, 'avg_confidence': float, 'frames_present': int}}
                - 'detected_classes': set of class names found
                - 'missed_classes': all COCO classes not detected
        """
        per_frame = []
        class_stats: dict[str, dict] = {}

        for t in range(frames.shape[0]):
            dets = self.detect_frame(frames[t])
            per_frame.append(dets)
            for d in dets:
                name = d["label"]
                if name not in class_stats:
                    class_stats[name] = {"count": 0, "total_conf": 0.0, "frames_present": set()}
                class_stats[name]["count"] += 1
                class_stats[name]["total_conf"] += d["score"]
                class_stats[name]["frames_present"].add(t)

        summary = {}
        for name, stats in class_stats.items():
            summary[name] = {
                "count": stats["count"],
                "avg_confidence": stats["total_conf"] / stats["count"],
                "frames_present": len(stats["frames_present"]),
            }

        detected = set(summary.keys())
        all_classes = set(self.class_names[1:])  # skip background
        missed = all_classes - detected

        return {
            "per_frame": per_frame,
            "summary": summary,
            "detected_classes": detected,
            "missed_classes": missed,
            "total_frames": frames.shape[0],
        }
