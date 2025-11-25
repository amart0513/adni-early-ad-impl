import argparse, os, glob, numpy as np
from tensorflow.keras.models import load_model # type: ignore
import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf

def load_volume(path):
    vol = nib.load(path).get_fdata().astype('float32')
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    return np.expand_dims(vol, axis=(0,-1))

def grad_cam(model, volume, layer_name=None):
    if layer_name is None:
        for layer in reversed(model.layers):
            if 'conv3d' in layer.name:
                layer_name = layer.name
                break
    conv_layer = model.get_layer(layer_name)
    grad_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(volume)
        cls = tf.argmax(predictions[0])
        class_channel = predictions[:, cls]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2,3))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-8)
    return heatmap, int(cls)

def save_slice_grid(heatmap, out_path, n=9):
    D = heatmap.shape[0]
    idxs = np.linspace(0, D-1, n, dtype=int)
    fig, axes = plt.subplots(3,3, figsize=(6,6))
    for ax, z in zip(axes.flatten(), idxs):
        ax.imshow(heatmap[z,:,:], cmap='hot')
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preproc_dir', required=True)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--out_dir', default='figures/gradcam_panels')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = load_model(args.model_path, compile=False)
    files = glob.glob(os.path.join(args.preproc_dir, '*.nii*'))[:30]
    for f in files:
        vol = load_volume(f)
        heatmap, cls = grad_cam(model, vol)
        base = os.path.splitext(os.path.basename(f))[0].replace('.nii','')
        outp = os.path.join(args.out_dir, f'{base}_cls{cls}.png')
        save_slice_grid(heatmap, outp)

if __name__ == '__main__':
    main()
