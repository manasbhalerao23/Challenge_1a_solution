
# Challenge 1A – Understand Your Document  

##  Objective

Extract Title, H1, H2, and H3 headings from diverse travel-related PDFs using classical machine learning techniques in a fully offline environment.

---

##  Approach

1. **PDF Parsing**:
   - Use `PyMuPDF` (`fitz`) to extract text blocks, font sizes, styles, and positions from each page.

2. **Feature Engineering**:
   - Generate features like:
     - Font size
     - Font weight (bold/regular)
     - Uppercase ratio
     - Indentation
     - Line spacing
     - Text position on page
     - Word count and character count
     - For Japanese: segmentation features via `TinySegmenter` and `Jibea`

3. **Heading Classification**:
   - Used a trained `XGBoost` model to classify each text block into one of:
     - `Title`, `H1`, `H2`, `H3`

4. **Output**:
   - Filter and sort the predictions.
   - Save results in a structured `JSON` format with headings and their levels.

---

##  Model

- **Algorithm**: XGBoost Classifier
- **Trained on**: Hand-labeled heading data from sample travel PDFs
- **Inputs**: Feature vector extracted from PDF layout and text
- **Outputs**: Heading level classification (Title, H1, H2, H3)

Model files:
- `extractor/xgb_heading_classifier.pkl` – Trained model
- `extractor/label_encoder.pkl` – Encodes/decodes heading labels

---

##  Libraries Used

- `PyMuPDF (fitz)` – PDF parsing
- `xgboost` – Heading classification model
- `scikit-learn` – Label encoding and preprocessing
- `pandas` – Data manipulation
- `numpy` – Numerical operations
- `TinySegmenter` – Japanese segmentation
- `jibea` – BERT-based Japanese feature support
- `joblib` – Model serialization

---

##  How to Build and Run using Docker
### Prepare Input Folder

Before running the Docker container, create an `input/` folder in the project directory and add all the PDF files to be tested inside it.

### Build Docker Image

```bash
docker build --platform linux/amd64 -t challenge_1a:solutionidentifier .
```

### Run Docker Container

```bash
docker run --rm -v "$(pwd)/input":/app/input -v "$(pwd)/output":/app/output challenge_1a:solutionidentifier
```

---

##  Output Format

 The output will be in a output folder as output_docs.json.

Example:

```json
[
  {
    "heading": "Discover South France",
    "level": "Title",
    "page": 1
  },
  {
    "heading": "Top Attractions",
    "level": "H1",
    "page": 2
  },
  {
    "heading": "1. Nice",
    "level": "H2",
    "page": 3
  }
]
```

---