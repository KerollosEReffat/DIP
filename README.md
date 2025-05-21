# Image Processing Pipeline

A Python-based image processing pipeline that performs the following steps on an input image:

1. **Brightness & Contrast Adjustment**: Scales pixel values by a contrast factor (alpha) and adds a brightness offset (beta).
2. **Gaussian Blur**: Reduces noise and detail using a Gaussian filter with configurable kernel size.
3. **Non-Local Means Denoising**: Applies a denoising algorithm to remove color noise while preserving edges.
4. **Canny Edge Detection**: Extracts edges from the denoised image using adjustable thresholds.
5. **Feature Extraction**: Computes a color histogram for each channel and concatenates them into a feature vector.

---

## Repository Structure

```
project-root/
├── src/                           # The folder contains all the code files.
│   ├── image_pipeline.py          # core processing functions
│   ├── utils.py                   # file ops and helpers
│   └── main.py                    # CLI entry point
├── images/                        # The folder contains all images (Sample input image, Output images)
│   ├── input.jpg                  # Sample input image
│   └── results/                   # The results folders contain dataset file
│       ├── <Output folders>/      # Any number of Output folders contain images and CSV
│       └── dataset.csv            # dataset file contains all features from all input images
├── requirements.txt               # Python dependencies
└── README.md                      # Project overview and usage (this file)
```

---

## Prerequisites

* Python 3.6 or higher
* pip (Python package manager)

---

## Installation

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd project-root
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the program with arguments:

 - PowerShell:
```bash
python src/main.py `
  --input path/to/image.jpg `
  --alpha 1.3 --beta 30 `
  --ksize 7 7 `
  --low 40 --high 120 `
  --featfile features.csv `
  --normalize
```

 - Command Line or any Terminal:
```bash
python src/main.py \
  --input path/to/image.jpg \
  --alpha 1.3 --beta 30 \
  --ksize 7 7 \
  --low 40 --high 120 \
  --featfile features.csv \
  --normalize
```
#### Notes:
 - PowerShell (Windows):
   - `Use ``` to break lines.`
 - Command Line (Windows) / Bash (Linux, macOS, Git Bash):
   - `Use \ to break lines.`<br/><br/>

| Argument     | Description                                | Default        |
| ------------ | ------------------------------------------ | -------------- |
| `--input`    | Path to the input image                    | **(required)** |
| `--alpha`    | Contrast adjustment factor                 | `1.2`          |
| `--beta`     | Brightness offset                          | `20`           |
| `--ksize`    | Gaussian blur kernel size (width height)   | `5 5`          |
| `--low`      | Canny edge detection low threshold         | `50`           |
| `--high`     | Canny edge detection high threshold        | `150`          |
| `--featfile` | CSV file name to save feature vectors      | `features.csv` |
| `--normalize`| Apply L2 normalization to feature vectors  | `False`        |

---

## Output

After running the script, `images/results/` will contain:

* `0_original.jpg`
* `1_brightness.jpg`
* `2_blur.jpg`
* `3_denoise.jpg`
* `4_edges.jpg`
* `features.csv`

Each step image is saved in sequence, and the final CSV includes a header row followed by one row per processed image.

---

## Customization

* Modify default parameters directly via CLI flags for flexible experimentation.
* Integrate additional processing functions by adding new modules in `src/` and calling them in the `main()` function.
* Extend feature extraction (e.g., HOG, SIFT) by updating the `extract_features` function.

---

## Contributing

Feel free to submit issues or pull requests for improvements and new features.

---

## License

This project is released under the MIT License.
