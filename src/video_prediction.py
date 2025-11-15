from ultralytics import YOLO
import cv2
import argparse
import os

# Class indices from your dataset
HAT_CLASS_ID = 0       # 'hat'
PERSON_CLASS_ID = 1    # 'person' (no helmet)


def load_model(model_path: str) -> YOLO:
    """Load YOLO model from file."""
    return YOLO(model_path)


def process_frame(model: YOLO, frame, conf: float = 0.25):
    """
    Detect helmets and persons in a single video frame.
    Draw ONLY green/red bounding boxes (NO TEXT) and return the annotated frame.
    """
    h_img, w_img = frame.shape[:2]

    # Scale line thickness based on frame size (consistent look)
    base = max(h_img, w_img) / 640.0
    box_thickness = max(2, int(4 * base))

    # Run YOLO model on the frame
    results = model.predict(frame, conf=conf, verbose=False)[0]

    if results.boxes is None or results.boxes.xyxy.shape[0] == 0:
        return frame  # no detections, return frame as-is

    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    for box, cls_id in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)

        # Helmet → green box
        if cls_id == HAT_CLASS_ID:
            color = (0, 255, 0)

        # No helmet → red box
        elif cls_id == PERSON_CLASS_ID:
            color = (0, 0, 255)

        # Any unexpected class (should not occur)
        else:
            color = (255, 255, 255)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            box_thickness
        )

    return frame


def process_video(model: YOLO, input_path: str, output_path: str, conf: float = 0.25):
    """
    Run helmet/no-helmet detection on a video file and save an annotated video.
    """

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {input_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # VideoWriter fourcc and output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'XVID', 'avc1', etc.
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        annotated = process_frame(model, frame, conf=conf)
        out.write(annotated)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"[INFO] Processed {frame_idx} frames...")

    cap.release()
    out.release()
    print(f"[INFO] Saved annotated video to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Helmet detection on video with colored boxes only."
    )
    parser.add_argument("model", help="Path to YOLOv11 model (.pt)")
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("output", help="Output video file path (e.g., out.mp4)")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    args = parser.parse_args()

    model = load_model(args.model)
    process_video(model, args.input, args.output, conf=args.conf)


if __name__ == "__main__":
    main()
