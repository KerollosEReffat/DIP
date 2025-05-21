import argparse
import logging 
import os
import cv2
import shutil

from image_pipeline import (
    adjust_brightness_contrast,
    apply_gaussian_blur,
    denoise,
    detect_edges,
    extract_features,
    build_dataset
)
from utils import load_image, ensure_dir, save_features

# configure root logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
logger = logging.getLogger()

def parse_args():
    parser = argparse.ArgumentParser(description='Image processing pipeline')
    parser.add_argument('--input', required=True, help='Path to input image')
    parser.add_argument('--alpha', type=float, default=1.2, help='Contrast factor')
    parser.add_argument('--beta', type=int, default=20, help='Brightness offset')
    parser.add_argument('--ksize', type=int, nargs=2, default=[5,5], help='Gaussian kernel size')
    parser.add_argument('--low', type=int, default=50, help='Canny low threshold')
    parser.add_argument('--high', type=int, default=150, help='Canny high threshold')
    parser.add_argument('--featfile', default='features.csv', help='CSV for feature vectors')
    parser.add_argument('--normalize', action='store_true', help='Apply L2 normalization to feature vectors')
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        logger.error(f"Input file not found: \"{args.input}\"")
        return

    output_path = "images/results"
    image_name = os.path.splitext(os.path.basename(args.input))[0]
    dataset_path = os.path.join(output_path, "dataset.csv")

    image_path = os.path.join(output_path, image_name)
    ensure_dir(image_path)

    # extract file extension (e.g. ".jpg")
    extension = os.path.splitext(args.input)[1]

    # prepare destination path with new name
    dst_path = os.path.join(image_path, f"0_original{extension}")

    # copy the original image to output directory with the new name
    shutil.copy(args.input, dst_path)
    logging.info(f"Copied original image.")

    img = load_image(args.input)
    if img is None: return

    bright = adjust_brightness_contrast(img, args.alpha, args.beta)
    cv2.imwrite(os.path.join(image_path, '1_brightness.jpg'), bright)
    logger.info('Brightness adjusted saved.')

    blurred = apply_gaussian_blur(bright, args.ksize)
    cv2.imwrite(os.path.join(image_path, '2_blur.jpg'), blurred)
    logger.info('Blurred image saved.')

    denoised = denoise(blurred)
    cv2.imwrite(os.path.join(image_path, '3_denoise.jpg'), denoised)
    logger.info('Denoised image saved.')

    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    edges = detect_edges(gray, args.low, args.high)
    cv2.imwrite(os.path.join(image_path, '4_edges.jpg'), edges)
    logger.info('Edges detected saved.')

    features = extract_features(denoised)
    save_features(features, os.path.join(image_path, args.featfile))
    logger.info('Features saved to Local CSV.')

    build_dataset(features, dataset_path, image_name, image_path, normalize=args.normalize)
    logger.info(f'Features saved to Big CSV (DataSet) with Name : \"{image_name}\" and \n\tPath : \"{os.path.abspath(dst_path)}\".')

    logger.info(f"Done, All files saved to \n\tFolder Path : \"{os.path.abspath(image_path)}\" \n\tand Big CSV (DataSet) Path : \"{os.path.abspath(dataset_path)}\".")

if __name__ == '__main__':
    main()