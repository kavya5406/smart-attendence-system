import cv2
import numpy as np
from pathlib import Path


def generate_synthetic_face(size=(128, 128), seed=None):
    if seed is not None:
        np.random.seed(seed)
    img = np.ones((*size, 3), dtype=np.uint8) * 200

    cx, cy = size[1] // 2, size[0] // 2
    axes = (size[1] // 3, size[0] // 2)
    cv2.ellipse(img, (cx, cy), axes, 0, 0, 360, (180, 150, 120), -1)

    eye_offset_x = size[1] // 8
    eye_y = cy - size[0] // 8
    eye_r = size[0] // 25
    cv2.circle(img, (cx - eye_offset_x, eye_y), eye_r, (50, 50, 50), -1)
    cv2.circle(img, (cx + eye_offset_x, eye_y), eye_r, (50, 50, 50), -1)

    cv2.circle(img, (cx - eye_offset_x, eye_y), eye_r // 3, (255, 255, 255), -1)
    cv2.circle(img, (cx + eye_offset_x, eye_y), eye_r // 3, (255, 255, 255), -1)

    nose_y = cy + size[0] // 12
    cv2.ellipse(img, (cx, nose_y), (size[1] // 12, size[0] // 15), 0, 0, 360, (100, 80, 60), -1)

    mouth_y = cy + size[0] // 4
    cv2.ellipse(img, (cx, mouth_y), (size[1] // 6, size[0] // 20), 0, 0, 180, (60, 40, 40), 2)

    noise = np.random.randint(-20, 20, img.shape, dtype=np.int8)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def generate_dataset(base_dir="data/raw", num_students=5, images_per_student=10):
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic dataset for {num_students} students...")
    for student_id in range(1, num_students + 1):
        sid = f"S{student_id:04d}"
        student_dir = base / sid
        student_dir.mkdir(parents=True, exist_ok=True)

        for i in range(images_per_student):
            variations = [
                {"seed": student_id * 1000 + i, "desc": "neutral"},
                {"seed": student_id * 1000 + i + 5000, "desc": "lighting"},
                {"seed": student_id * 1000 + i + 10000, "desc": "expression"},
            ]
            for j, var in enumerate(variations):
                img = generate_synthetic_face(seed=var["seed"])
                cv2.imwrite(
                    str(student_dir / f"{var['desc']}_{i:03d}.jpg"), img
                )

        print(f"  {sid}: {images_per_student * 3} images generated")

    print(f"Dataset generated at {base.resolve()}")
    print(f"Total students: {num_students}")
    print(f"Total images: {num_students * images_per_student * 3}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--students", type=int, default=5)
    parser.add_argument("--images", type=int, default=10)
    parser.add_argument("--output", type=str, default="data/raw")
    args = parser.parse_args()
    generate_dataset(args.output, args.students, args.images)
