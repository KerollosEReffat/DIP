import os
from logging import getLogger

logger = getLogger(__name__)

def load_image(path):
    """Read image from disk; raise if missing."""
    from cv2 import imread
    img = imread(path)
    if img is None:
        logger.error(f"Image not found: {path}")
        raise FileNotFoundError(f"Cannot load: {path}")
    return img


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_features(feat_vec, filepath):
    """
    Save feature vector to CSV file as two columns: 'bin' and 'value'.
    """
    # prepare list of rows: each [bin_name, value]
    rows = []
    for i, val in enumerate(feat_vec.flatten()):
        # construct row label and formatted value
        rows.append([f'bin{i}', f'{val:.6f}'])

    from csv import writer
    # open (or create) CSV and write header+rows
    with open(filepath, mode='w', newline='') as file:
        write = writer(file)
        # write header row
        write.writerow(['bin', 'value'])
        # write data rows
        write.writerows(rows)