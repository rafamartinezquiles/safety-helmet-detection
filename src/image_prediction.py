from ultralytics import YOLO
import cv2
import os
import argparse

# Class indices from your dataset
HAT_CLASS_ID = 0       # 'hat'
PERSON_CLASS_ID = 1    # 'person' (no helmet)


def load_model(model_path: str) -> YOLO:
    """Load YOLO model from file."""
    return YOLO(model_path)


def process_image(model: YOLO, image_path: str, output_path: str, conf: float = 0.25):
    """
    Detect helmets and persons in an image.
    Draw ONLY green/red bounding boxes (NO TEXT).
    """

    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Could not read: {image_path}")
        return

    h_img, w_img = img.shape[:2]

    # Scale line thickness based on image size (consistent look)
    base = max(h_img, w_img) / 640.0
    box_thickness = max(2, int(4 * base))  # same for all boxes

    # Run YOLO model
    results = model.predict(img, conf=conf, verbose=False)[0]

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

        # Should not occur (nc=2)
        else:
            color = (255, 255, 255)

        # Draw bounding box (NO TEXT)
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color,
            box_thickness
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"[INFO] Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Helmet detection with colored boxes only.")
    parser.add_argument("model", help="Path to YOLOv11 model (.pt)")
    parser.add_argument("input", help="Image file OR folder")
    parser.add_argument("output", help="Output image OR folder")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")

    args = parser.parse_args()
    model = load_model(args.model)

    # Folder mode
    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        for fname in os.listdir(args.input):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                in_path = os.path.join(args.input, fname)
                out_path = os.path.join(args.output, fname)
                process_image(model, in_path, out_path, args.conf)

    # Single image mode
    else:
        process_image(model, args.input, args.output, args.conf)


if __name__ == "__main__":
    main()
