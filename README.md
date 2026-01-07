# AI Watermark Remover

A powerful, automated tool to remove text watermarks and unwanted text overlays from images. It combines **EasyOCR** for robust text detection and **LaMa (Large Mask Inpainting)** for state-of-the-art background reconstruction.

## 🚀 Features

*   **Dual-Pass Detection**: Scans both the original and a contrast-enhanced (CLAHE) version of the image to detect faint or hidden watermarks.
*   **Dynamic Masking**: Automatically adjusts the inpainting mask size based on the text height—thicker for large headers, thinner for fine print—preserving image details.
*   **State-of-the-Art Inpainting**: Uses the **LaMa** model to seamlessly fill in the removed text areas, often producing results indistinguishable from the original background.
*   **Batch Processing**: Processes all images in the `inputs/` directory automatically.
*   **GPU Support**: Utilizes NVIDIA CUDA if available for faster processing (EasyOCR), with automatic fallback to CPU.

## 🛠️ Installation

1.  **Clone the repository** (or download the source code):
    ```bash
    git clone <your-repo-url>
    cd <your-repo-folder>
    ```

2.  **Install Dependencies**:
    It is recommended to use a virtual environment (venv or conda).
    ```bash
    pip install -r requirements.txt
    ```

    *Prerequisites*:
    *   Python 3.8+
    *   `opencv-python`
    *   `easyocr`
    *   `numpy`
    *   `simple-lama-inpainting`
    *   `pillow`

## 📖 Usage

1.  **Prepare your images**:
    Place all the images you want to clean inside the `inputs` folder. If the folder doesn't exist, run the script once and it will create it for you.
    
    *Supported formats: .jpg, .jpeg, .png, .bmp, .webp*

2.  **Run the script**:
    ```bash
    python main.py
    ```

3.  **Check results**:
    The cleaned images will be saved in the `outputs` directory.

4.  **(Optional) Test Run**:
    You can generate a sample test image to verify functionality:
    ```bash
    python create_test_image.py
    python main.py
    ```

## 📂 Project Structure

```
├── inputs/                 # Place input images here
├── outputs/                # Processed images appear here
├── main.py                 # Core logic: detection + inpainting
├── create_test_image.py    # Helper script to create a dummy test image
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## ⚙️ How It Works

1.  **Load Image**: Reads the input file.
2.  **Preprocessing**: Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to create a high-contrast version of the image.
3.  **Detection**: Runs EasyOCR on both the original and the enhanced image to find text bounding boxes.
4.  **Mask Generation**: Merges detections and creates a binary mask. It calculates the height of each text block and dynamically dilates the mask to ensure full coverage without over-masking.
5.  **Inpainting**: Feeds the image and the mask into the LaMa deep learning model to reconstruct the missing pixels.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
