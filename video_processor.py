"""
===============================================================================
Video Dehazing & Temporal Smoothing Module
===============================================================================
Applies frame-by-frame deep neural dehazing on video streams (MP4/AVI/Webcam)
with Temporal Exponential Moving Average (EMA) smoothing to eliminate flickering.
"""

import os
import time
from typing import Optional, Dict, Any, Callable
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from model import DAM_RFRNet

class VideoDehazeProcessor:
    """
    Video Dehazing engine managing frame extraction, neural model inference,
    temporal consistency filtering, and video encoding.
    """
    def __init__(self, model_checkpoint: Optional[str] = "./checkpoints/dam_rfrnet_best.pth", smoothing_alpha: float = 0.7):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.smoothing_alpha = smoothing_alpha
        self.model = DAM_RFRNet().to(self.device)
        self._load_weights(model_checkpoint)
        self.model.eval()
        
        self.to_tensor = transforms.ToTensor()

    def _load_weights(self, checkpoint_path: Optional[str]):
        """Loads trained DAM-RFRNet model weights if available."""
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
                self.model.load_state_dict(state_dict)
                print(f"[+] Loaded video dehazer model weights from {checkpoint_path}")
            except Exception as e:
                print(f"[!] Warning: Could not load video weights ({e}). Using initialized weights.")

    @torch.no_grad()
    def process_frame(self, frame_bgr: np.ndarray, prev_frame_float: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dehazes a single OpenCV BGR video frame and applies temporal exponential smoothing.
        Returns: (Processed BGR uint8 frame, Raw float frame for temporal state)
        """
        # Convert BGR (OpenCV) -> RGB PIL Image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Tensor preparation
        tensor_in = self.to_tensor(pil_img).unsqueeze(0).to(self.device)
        
        # Model forward pass
        raw_out_tensor = self.model(tensor_in)
        raw_out_tensor = torch.clamp(raw_out_tensor.squeeze(0).cpu(), 0.0, 1.0)
        
        # Convert back to HWC numpy float array [0.0, 1.0] in RGB
        curr_frame_float = raw_out_tensor.permute(1, 2, 0).numpy()
        
        # Temporal Exponential Smoothing: F_smooth = alpha * F_curr + (1 - alpha) * F_prev
        if prev_frame_float is not None and prev_frame_float.shape == curr_frame_float.shape:
            smoothed_float = self.smoothing_alpha * curr_frame_float + (1.0 - self.smoothing_alpha) * prev_frame_float
        else:
            smoothed_float = curr_frame_float

        # Convert back to BGR uint8 [0, 255]
        smoothed_rgb = np.uint8(np.clip(smoothed_float * 255.0, 0, 255))
        output_bgr = cv2.cvtColor(smoothed_rgb, cv2.COLOR_RGB2BGR)
        
        return output_bgr, smoothed_float

    def process_video_file(
        self,
        input_video_path: str,
        output_video_path: str = "./outputs/dehazed_video.mp4",
        progress_callback: Optional[Callable[[float, int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Processes an input video file and writes the dehazed video to output_video_path.
        """
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        cap = cv2.VideoCapture(input_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        # Video writer codec selection
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        prev_state = None
        processed_count = 0
        start_time = time.perf_counter()

        print(f"[*] Processing Video: {input_video_path} ({width}x{height} @ {fps} FPS, {total_frames} frames)")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            out_frame, prev_state = self.process_frame(frame, prev_state)
            out_writer.write(out_frame)

            processed_count += 1
            if progress_callback and total_frames > 0:
                progress_callback(processed_count / total_frames, processed_count, total_frames)

        cap.release()
        out_writer.release()

        elapsed = time.perf_counter() - start_time
        processing_fps = processed_count / elapsed if elapsed > 0 else 0

        stats = {
            'input_path': input_video_path,
            'output_path': output_video_path,
            'total_frames': processed_count,
            'elapsed_time_sec': elapsed,
            'processing_fps': processing_fps
        }
        print(f"[+] Video processing complete! Output saved to: {output_video_path}")
        return stats


if __name__ == '__main__':
    # Generator for synthetic test video
    h, w = 240, 320
    test_video = "./outputs/test_hazy.mp4"
    os.makedirs("./outputs", exist_ok=True)
    
    writer = cv2.VideoWriter(test_video, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
    for _ in range(15):
        frame = np.uint8(np.random.rand(h, w, 3) * 200 + 55) # synthetic hazy frame
        writer.write(frame)
    writer.release()

    processor = VideoDehazeProcessor()
    res = processor.process_video_file(test_video, "./outputs/test_dehazed.mp4")
    print(f"[+] Video Processor Test Success! Stats: {res}")
