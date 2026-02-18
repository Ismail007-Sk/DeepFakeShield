from pathlib import Path
import cv2
import os

# ---------- PATHS ----------
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

RAW_REAL = ROOT_DIR / "RawDataset" / "Real"
RAW_FAKE = ROOT_DIR / "RawDataset" / "Fake"

PRE_REAL = BASE_DIR / "Dataset" / "Real"
PRE_FAKE = BASE_DIR / "Dataset" / "Fake"

# ---------- SETTINGS ----------
IMG_SIZE = 224
FRAME_SKIP = 90   # take every 10th frame from video

# ---------- CREATE OUTPUT FOLDERS ----------
PRE_REAL.mkdir(parents=True, exist_ok=True)
PRE_FAKE.mkdir(parents=True, exist_ok=True)


def process_image(img_path, output_path, counter, class_name):
    img = cv2.imread(str(img_path))
    if img is None:
        return counter

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    save_path = output_path / f"{class_name}_{counter}.jpg"
    cv2.imwrite(str(save_path), img)
    return counter + 1


def process_video(video_path, output_path, counter, class_name):
    cap = cv2.VideoCapture(str(video_path))
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % FRAME_SKIP == 0:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            save_path = output_path / f"{class_name}_{counter}.jpg"
            cv2.imwrite(str(save_path), frame)
            counter += 1

        frame_id += 1

    cap.release()
    return counter


def preprocess_and_save(input_path, output_path, class_name):
    counter = 0

    for file in os.listdir(input_path):
        file_path = input_path / file
        suffix = file_path.suffix.lower()

        if suffix in [".jpg", ".jpeg", ".png", ".bmp", ".webp",".heic"]:
            counter = process_image(file_path, output_path, counter, class_name)

        elif suffix in [".mp4", ".avi", ".mov", ".mkv"]:
            counter = process_video(file_path, output_path, counter, class_name)

    print(f"Processed {counter} samples for {class_name}")



# ---------- RUN ----------
if __name__ == "__main__":
    print("Starting preprocessing (Image + Video) for DeepFakeShield...")

    preprocess_and_save(RAW_REAL, PRE_REAL, "Real")
    preprocess_and_save(RAW_FAKE, PRE_FAKE, "Fake")

    print("Preprocessing completed successfully ✅")
