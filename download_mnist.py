import urllib.request, gzip, shutil, os

BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "mnistrain",
    "train-labels-idx1-ubyte.gz": "mnilabel",
    "t10k-images-idx3-ubyte.gz":  "10k_test_images",
    "t10k-labels-idx1-ubyte.gz":  "10k_test_labels",
}
for src, dst in FILES.items():
    if os.path.exists(dst):
        continue
    print("downloading", src)
    urllib.request.urlretrieve(BASE + src, src)
    with gzip.open(src, "rb") as f_in, open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(src)
print("done")
