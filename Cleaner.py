import cv2
import os

dataset_path = r"C:\Users\Desktop\PycharmProjects\DeepFakeShield\RawDataset\Fake"  # 🔁 Change if needed

deleted_files = []
video_checked = 0
image_checked = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        file_path = os.path.join(root, file)

        # -------------------- VIDEO CHECK --------------------
        if file.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            video_checked += 1
            cap = cv2.VideoCapture(file_path)

            if not cap.isOpened():
                print("Deleting corrupted video (cannot open):", file_path)
                cap.release()
                os.remove(file_path)
                deleted_files.append(file_path)
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            read_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                read_frames += 1

            cap.release()

            # If video stopped early OR no frames read
            if total_frames == 0 or read_frames < 0.9 * total_frames:
                print("Deleting corrupted video (early stop):", file_path)
                os.remove(file_path)
                deleted_files.append(file_path)

        # -------------------- IMAGE CHECK --------------------
        elif file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            image_checked += 1
            img = cv2.imread(file_path)

            if img is None or img.size == 0:
                print("Deleting corrupted image:", file_path)
                os.remove(file_path)
                deleted_files.append(file_path)

# -------------------- SUMMARY --------------------
print("\n========== CLEANING SUMMARY ==========")
print("Videos Checked:", video_checked)
print("Images Checked:", image_checked)
print("Total Deleted:", len(deleted_files))
print("======================================")
