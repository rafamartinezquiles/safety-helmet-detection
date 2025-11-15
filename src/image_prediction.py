from ultralytics import YOLO
import cv2
import os
import argparse


# Class indices from your YAML
HAT_CLASS_ID = 0       # 'hat'
PERSON_CLASS_ID = 1    # 'person' (only when there is NO helmet)


def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def process_image(model: YOLO, image_path: str, output_path: str, conf: float = 0.25):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Could not read image: {image_path}")
        return

    # Run YOLOv11
    results = model.predict(img, conf=conf, verbose=False)[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    for box, cls_id in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)

        if cls_id == PERSON_CLASS_ID:
            # Person without helmet -> RED
            color = (0, 0, 255)  # BGR
            label = "NO HELMET"
        elif cls_id == HAT_CLASS_ID:
            # Helmet -> GREEN
            color = (0, 255, 0)
            label = "HELMET"
        else:
            # Any unexpected class (shouldn't happen with nc=2)
            color = (255, 255, 255)
            label = str(cls_id)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"[INFO] Saved result to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Highlight people without helmets in red and helmets in green."
    )
    parser.add_argument("model", help="Path to YOLOv11 model (.pt)")
    parser.add_argument("input", help="Image file OR folder with images")
    parser.add_argument("output", help="Output image file OR folder")
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)"
    )
    args = parser.parse_args()

    model = load_model(args.model)

    if os.path.isdir(args.input):
        # Folder mode
        os.makedirs(args.output, exist_ok=True)
        for fname in os.listdir(args.input):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                in_path = os.path.join(args.input, fname)
                out_path = os.path.join(args.output, fname)
                process_image(model, in_path, out_path, conf=args.conf)
    else:
        # Single image mode
        process_image(model, args.input, args.output, conf=args.conf)


if __name__ == "__main__":
    main()
