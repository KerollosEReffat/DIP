# import necessary modules
import os
# import OpenCV for image operations
import cv2
# import NumPy for array and histogram handling
import numpy as np
# import argparse for command-line interface
import argparse
# import logging for status messages and debugging
import logging

# set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_args():
    """
    Parse command-line arguments.
    """
    # create argument parser
    parser = argparse.ArgumentParser(description='Image processing pipeline')
    # input image path argument
    parser.add_argument('--input', required=True, help='Path to input image')
    # output directory argument
    parser.add_argument('--outdir', default='results', help='Directory to save outputs')
    # contrast factor argument
    parser.add_argument('--alpha', type=float, default=1.2, help='Contrast adjustment factor')
    # brightness offset argument
    parser.add_argument('--beta', type=int, default=20, help='Brightness adjustment offset')
    # Gaussian blur kernel size argument
    parser.add_argument('--ksize', type=int, nargs=2, default=[5,5], help='Gaussian blur kernel size')
    # low threshold for Canny edge detection
    parser.add_argument('--low', type=int, default=50, help='Canny low threshold')
    # high threshold for Canny edge detection
    parser.add_argument('--high', type=int, default=150, help='Canny high threshold')
    # features output file argument
    parser.add_argument('--featfile', default='features.csv', help='CSV file to save feature vectors')
    # return parsed arguments
    return parser.parse_args()

def load_image(path):
    """
    Load an image from disk and validate.
    """
    # read image from path
    img = cv2.imread(path)
    # if read failed, log and raise
    if img is None:
        logging.error(f'Cannot load image: {path}')
        raise FileNotFoundError(f'Image not found: {path}')
    # return loaded image
    return img

def adjust_brightness_contrast(img, alpha, beta):
    """
    Adjust image brightness and contrast.
    """
    # scale pixel values by alpha and add beta
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def apply_gaussian_blur(img, ksize):
    """
    Apply Gaussian blur to reduce noise/detail.
    """
    # blur with specified kernel size
    return cv2.GaussianBlur(img, tuple(ksize), 0)

def denoise(img):
    """
    Perform non-local means denoising on color image.
    """
    # parameters: hColor=10, h=10, templateWindow=7, searchWindow=21
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

def detect_edges(img_gray, low, high):
    """
    Detect edges using Canny algorithm on grayscale image.
    """
    # run Canny edge detector
    return cv2.Canny(img_gray, low, high)

def extract_features(img):
    """
    Extract a color histogram feature vector.
    """
    # split into color channels
    chans = cv2.split(img)
    # list to collect histograms
    hist = []
    # compute histogram for each channel
    for c in chans:
        h = cv2.calcHist([c], [0], None, [256], [0,256]).flatten()
        hist.append(h)
    # concatenate into single vector
    return np.concatenate(hist)

def save_features(feat_vec, filepath):
    """
    Save feature vector to CSV file.
    """
    # if file not exists, write header
    header = ','.join([f'bin{i}' for i in range(len(feat_vec))])
    # save with header if new file
    if not os.path.exists(filepath):
        np.savetxt(filepath, feat_vec.reshape(1, -1), delimiter=',', header=header, comments='')
    else:
        # append without header
        np.savetxt(filepath, feat_vec.reshape(1, -1), delimiter=',', comments='')

def ensure_dir(path):
    """
    Create directory if it does not exist.
    """
    # make directories as needed
    os.makedirs(path, exist_ok=True)

def main():
    """
    Main pipeline execution.
    """
    # parse CLI arguments
    args = parse_args()
    # ensure output directory exists
    ensure_dir(args.outdir)
    # load the input image
    img = load_image(args.input)

    # step 1: adjust brightness and contrast
    bright = adjust_brightness_contrast(img, args.alpha, args.beta)
    cv2.imwrite(os.path.join(args.outdir, '1_brightness.jpg'), bright)
    logging.info('Saved brightness-adjusted image')

    # step 2: apply Gaussian blur
    blurred = apply_gaussian_blur(bright, args.ksize)
    cv2.imwrite(os.path.join(args.outdir, '2_blur.jpg'), blurred)
    logging.info('Saved blurred image')

    # step 3: denoise the image
    denoised = denoise(blurred)
    cv2.imwrite(os.path.join(args.outdir, '3_denoise.jpg'), denoised)
    logging.info('Saved denoised image')

    # step 4: detect edges on grayscale
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    edges = detect_edges(gray, args.low, args.high)
    cv2.imwrite(os.path.join(args.outdir, '4_edges.jpg'), edges)
    logging.info('Saved edge-detected image')

    # step 5: extract features and save
    features = extract_features(denoised)
    save_features(features, os.path.join(args.outdir, args.featfile))
    logging.info('Feature vector saved to CSV')

if __name__ == '__main__':
    # run the main function
    main()
