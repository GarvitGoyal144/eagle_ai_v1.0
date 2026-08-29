import os
import cv2
import numpy as np
from typing import List, Dict, Any

from app.config.settings import settings

# Thread limits — safe for Render's shared CPU
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

# COCO class names (80 classes) — same as YOLO default training set
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


class DetectionService:
    """
    Memory-efficient YOLO object detection using ONNX Runtime.
    Uses ~50MB RAM instead of ~300MB with PyTorch — safe for Render 512MB free tier.
    """

    def __init__(self):
        self._session = None
        self._input_name = None
        self._infer_size = settings.INFERENCE_SIZE
        self._conf_thresh = settings.DETECTION_CONF

    def _ensure_loaded(self):
        if self._session is not None:
            return

        import onnxruntime as ort

        model_path = settings.YOLO_MODEL
        # If .pt path given, swap to .onnx equivalent
        if model_path.endswith(".pt"):
            model_path = model_path.replace(".pt", ".onnx")

        print(f"Loading YOLO ONNX ({model_path}) via onnxruntime...", flush=True)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        print(f"YOLO ONNX loaded successfully ✅ (~50MB RAM, no PyTorch)", flush=True)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalize, and format frame for ONNX inference."""
        img = cv2.resize(frame, (self._infer_size, self._infer_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))   # HWC → CHW
        img = np.expand_dims(img, 0)          # add batch dim → [1, 3, H, W]
        return img

    def _postprocess(self, output: np.ndarray) -> List[Dict[str, Any]]:
        """
        Parse YOLO ONNX output → detections with NMS.
        YOLO output shape: [1, 84, 8400]  (4 box coords + 80 class scores)
        """
        preds = output[0]        # [84, 8400]
        preds = preds.T          # [8400, 84]

        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:]

        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Filter by confidence threshold
        mask = max_scores >= self._conf_thresh
        if not np.any(mask):
            return []

        boxes_cxcywh = boxes_cxcywh[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        # Convert cx,cy,w,h → x1,y1,x2,y2
        cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        # NMS via OpenCV (no torch needed)
        nms_indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            max_scores.tolist(),
            self._conf_thresh,
            0.45,
        )

        detections = []
        for i, idx in enumerate(nms_indices):
            cls_id = int(class_ids[idx])
            cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
            detections.append({
                "track_id": i + 1,
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": float(max_scores[idx]),
                "bbox": boxes_xyxy[idx].tolist(),
            })

        return detections

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in a single frame."""
        self._ensure_loaded()
        inp = self._preprocess(frame)
        outputs = self._session.run(None, {self._input_name: inp})
        return self._postprocess(outputs[0])

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Detect objects in a list of frames.
        Runs each frame individually (ONNX CPU doesn't benefit much from batching).
        """
        self._ensure_loaded()
        return [self.detect(frame) for frame in frames]


detection_service = DetectionService()
