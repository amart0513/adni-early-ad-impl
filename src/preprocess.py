import os, argparse, glob
import numpy as np
import nibabel as nib
from skimage.transform import resize
from utils import ensure_dir

def normalize_01(vol):
    vmin, vmax = np.percentile(vol, 1), np.percentile(vol, 99)
    vol = np.clip(vol, vmin, vmax)
    vol = (vol - vmin) / (vmax - vmin + 1e-8)
    return vol

def preprocess_one(path, out_path, target_size=128):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    data = normalize_01(data)
    data_rs = resize(data, (target_size, target_size, target_size), mode="constant", anti_aliasing=True)
    out_img = nib.Nifti1Image(data_rs, affine=img.affine)
    nib.save(out_img, out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--target_size", type=int, default=128)
    args = ap.parse_args()

    ensure_dir(args.output_dir)
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.nii*")))
    if not files:
        print("No NIfTI files found. Place .nii or .nii.gz in data/raw.")
        return

    for f in files:
        base = os.path.basename(f)
        out_f = os.path.join(args.output_dir, base)
        preprocess_one(f, out_f, target_size=args.target_size)
        print("Preprocessed:", base)

if __name__ == "__main__":
    main()
