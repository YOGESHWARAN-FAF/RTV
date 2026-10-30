import gdown
import os


def download_gdrive_file(path, id):
    output = path
    if os.path.exists(path):
        print('File already exist, stop downloading.')
        return
    else:
        print("Downloading checkpoints...")
    tdir, _ = os.path.split(path)
    print(path)
    print(tdir)
    os.makedirs(tdir, exist_ok=True)
    gdown.download(
        f"https://drive.google.com/uc?export=download&confirm=pbef&id=" + id,
        output
    )
