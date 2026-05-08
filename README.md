# FYP
Geotagging Final Year Project
# Running the Project

## Prerequisites

Make sure the following are installed on your system:

- Python 3.9 or above
- Jupyter Notebook

Download Python from the official website:

https://www.python.org/downloads/

---

## Step 1: Clone the Repository

```bash
git clone <your-repository-link>
cd <repository-folder>
```

---

## Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

### Activate the Environment

#### Windows
```bash
venv\Scripts\activate
```

#### macOS/Linux
```bash
source venv/bin/activate
```

---

## Step 3: Install Required Dependencies

```bash
pip install torch ultralytics exifread pyproj jupyter notebook
```

---

## Step 4: Launch Jupyter Notebook

```bash
jupyter notebook
```

This will open Jupyter Notebook in your browser.

---

## Step 5: Open the Notebook

Open the following file:

```text
FYP1.ipynb
```

---

## Step 6: Run the Notebook

Run all notebook cells sequentially:

- Click **Kernel → Restart & Run All**
- OR use `Shift + Enter` for each cell

---

# Additional Notes

## Required Files

Before running the notebook, ensure that:

- YOLO `.pt` model files are available
- Input images/videos are placed in the correct directories
- Output folders referenced in the notebook exist

---

## GPU Support (Optional)

To check if GPU support is enabled:

```python
import torch
print(torch.cuda.is_available())
```

If the output is `True`, PyTorch is using GPU acceleration.

---

