import kagglehub
import os
import shutil

def download_and_prepare():
    print("[INFO] Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("phucthaiv02/butterfly-image-classification")

    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for file in ["Training_set.csv", "Testing_set.csv"]:
        shutil.copy(os.path.join(path, file), os.path.join(data_dir, file))
    shutil.copytree(os.path.join(path, "train"), os.path.join(data_dir, "train"), dirs_exist_ok=True)
    shutil.copytree(os.path.join(path, "test"), os.path.join(data_dir, "test"), dirs_exist_ok=True)

    print("[INFO] Dataset ready in ./data directory")

if __name__ == "__main__":
    download_and_prepare()
