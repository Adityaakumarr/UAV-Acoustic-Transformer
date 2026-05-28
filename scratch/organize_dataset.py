import os
import shutil
import zipfile

def organize_dataset():
    print("Starting dataset organization...")
    
    # 1. Move Mavic2.bag
    bag_src = "dataset/mavic2_dataset/Mavic2.bag"
    bag_dst = "dataset/bags/Mavic2.bag"
    if os.path.exists(bag_src):
        os.makedirs(os.path.dirname(bag_dst), exist_ok=True)
        shutil.move(bag_src, bag_dst)
        print(f"[OK] Moved Mavic2.bag to {bag_dst}")
    else:
        print(f"[WARN] {bag_src} not found (already moved?)")

    # 2. Extract Mavic2.zip
    zip_src = "dataset/mavic2_dataset/Mavic2.zip"
    extract_dst = "dataset/Mavic2"
    if os.path.exists(zip_src):
        os.makedirs(extract_dst, exist_ok=True)
        print(f"[WAIT] Extracting 12GB Mavic2.zip... This will take a few minutes.")
        with zipfile.ZipFile(zip_src, 'r') as zip_ref:
            zip_ref.extractall(extract_dst)
        print(f"[OK] Extraction complete to {extract_dst}")
    else:
        print(f"[WARN] {zip_src} not found (already extracted?)")

    # 3. Clean up mavic2_dataset
    if os.path.exists("dataset/mavic2_dataset"):
        shutil.rmtree("dataset/mavic2_dataset")
        print("[OK] Cleaned up old mavic2_dataset folder")

    # 4. Clean up pham4_dataset (the accidental duplicate)
    if os.path.exists("dataset/pham4_dataset"):
        shutil.rmtree("dataset/pham4_dataset")
        print("[OK] Cleaned up accidental pham4_dataset duplicate folder")

    print("\n[SUCCESS] Dataset structure is now perfect and ready for processing!")

if __name__ == "__main__":
    organize_dataset()

