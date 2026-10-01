import os
import sys
import urllib.request
import zipfile

URL = ("https://raw.githubusercontent.com/nmrksic/counter-fitting/"
       "master/word_vectors/counter-fitted-vectors.txt.zip")
# download next to this script, where pcls_mutation.py looks first
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(OUT_DIR, "counter-fitted-vectors.txt.zip")
OUT_PATH = os.path.join(OUT_DIR, "counter-fitted-vectors.txt")


def _progress(done, block, total):
    if total > 0:
        pct = min(100, done * block * 100 // total)
        sys.stdout.write(f"\r  downloading... {pct}%")
        sys.stdout.flush()


def main():
    if os.path.exists(OUT_PATH):
        print(f"{OUT_PATH} already exists; nothing to do.")
        return
    print(f"Fetching {URL}")
    urllib.request.urlretrieve(URL, ZIP_PATH, _progress)
    print("\n  unzipping...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        # the archive stores it as counter-fitted-vectors.txt
        z.extractall(OUT_DIR)
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    if not os.path.exists(OUT_PATH):
        # some mirrors nest the file; find and move it
        for root, _, files in os.walk(OUT_DIR):
            for f in files:
                if f == os.path.basename(OUT_PATH):
                    os.replace(os.path.join(root, f), OUT_PATH)
    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"  done -> {OUT_PATH} ({size_mb:.0f} MB)")
    print("Installation Successful")


if __name__ == "__main__":
    main()
