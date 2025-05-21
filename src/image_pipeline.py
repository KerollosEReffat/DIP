import cv2
from logging import getLogger

# configure logger for this module
logger = getLogger(__name__)

def adjust_brightness_contrast(img, alpha, beta):
    """Scale pixel values by alpha and add beta."""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def apply_gaussian_blur(img, ksize):
    """Reduce noise and detail via Gaussian blur."""
    return cv2.GaussianBlur(img, tuple(ksize), 0)

def denoise(img):
    """Perform non-local means denoising on a color image."""
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

def detect_edges(img_gray, low, high):
    """Apply Canny edge detector on grayscale image."""
    return cv2.Canny(img_gray, low, high)

def extract_features(img):
    """
    Compute concatenated color-histogram feature vector
    and return it as a vertical column vector.
    """
    # split image into its color channels
    chans = cv2.split(img)
    # list to accumulate each channel histogram
    hist = []
    for c in chans:
        # compute 256-bin histogram for channel c
        h = cv2.calcHist([c], [0], None, [256], [0,256]).flatten()
        hist.append(h)
    # concatenate all channel histograms and reshape to (N, 1)
    from numpy import concatenate
    return concatenate(hist).reshape(-1, 1)

def build_dataset(features, csv_path, image_name, image_path, normalize=False):
    """
    Build dataset CSV without external labels—use filename as label.
    """
    import csv, os, numpy as np

    # flatten features to 1D
    vec = features.flatten()

    # normalize if requested
    if normalize:
        norm = np.linalg.norm(vec)
        vec = vec / norm if norm > 0 else vec

    # prepare header and row
    header = [f'bin_{i}' for i in range(len(vec))] + ['image_name_(label)'] + ['image_link']
    row = vec.tolist() + [image_name] + [image_path]

    # write or append CSV
    exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)