import sys
import fitz  # PyMuPDF
import pandas as pd
import re
import string
import json
import joblib
import os
from pathlib import Path
from xgboost import XGBClassifier


INPUT_FOLDER = sys.argv[1] 
OUTPUT_FOLDER = sys.argv[2]  
MODEL_PATH = "extractor/xgb_heading_classifier.pkl"
ENCODER_PATH = "extractor/label_encoder.pkl"


FEATURE_COLUMNS = [
    'page', 'font_size', 'is_bold', 'x0', 'x1', 'top', 'width',
    'line_no', 'space_above', 'space_below',
    'word_count', 'is_uppercase_ratio', 'bottom',
    'num_count', 'punctuation', 'is_landscape'
]


def find_overlap_suffix_prefix(prev, curr):
    max_overlap = min(len(prev), len(curr))
    for i in range(max_overlap, 0, -1):
        if prev[-i:] == curr[:i]:
            return curr[i:]
    return curr


def clean_large_text(spans, size_threshold=30):
    cleaned = []
    buffer = []
    for span in spans:
        text = span["text"].strip()
        if not text:
            continue
        if span["size"] > size_threshold:
            if not buffer:
                buffer.append(span.copy())
            else:
                last = buffer[-1]
                last["text"] += find_overlap_suffix_prefix(last["text"], text)
        else:
            if buffer:
                cleaned.append(buffer[-1])
                buffer.clear()
            cleaned.append(span)
    if buffer:
        cleaned.append(buffer[-1])
    return cleaned


def extract_text_blocks(pdf_path, font_size_threshold=30):
    doc = fitz.open(pdf_path)
    rows = []

    for page_number in range(len(doc)):
        page = doc[page_number]
        page_rect = page.rect
        is_landscape = page_rect.width > page_rect.height
        
        spans = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span["page"] = page_number + 1
                    span["is_landscape"] = is_landscape
                    spans.append(span)

        spans = clean_large_text(spans, font_size_threshold)

        for i, span in enumerate(spans):
            text = span.get("text", "").strip()
            if not text:
                continue
            x0, y0, x1, y1 = span["bbox"]
            length = len(text)
            uppercase = sum(1 for c in text if c.isupper())

            rows.append({
                "group_id": None,
                "page": span["page"],
                "text": text,
                "font_size": round(span["size"], 3),
                "font_name": span.get("font", ""),
                "is_bold": "bold" in span.get("font", "").lower(),
                "x0": round(x0, 3),
                "x1": round(x1, 3),
                "top": round(y0, 3),
                "bottom": round(y1, 3),
                "width": round(x1 - x0, 3),
                "line_no": i + 1,
                "word_count": len(text.split()),
                "is_uppercase_ratio": round(uppercase / max(length, 1), 3),
                "is_landscape": span["is_landscape"],
                "label": None,
            })

    df = pd.DataFrame(rows).sort_values(by=["page", "top"]).reset_index(drop=True)

    
    df["space_above"] = (df["top"] - df["bottom"].shift(1)).fillna(0).round(3)
    df["space_below"] = (df["top"].shift(-1) - df["bottom"]).fillna(0).round(3)

    return df


def is_heading(row, font_size_threshold=13, uppercase_ratio_threshold=0.6):
    return row["is_bold"] and (
        row["font_size"] >= font_size_threshold
        or row["is_uppercase_ratio"] >= uppercase_ratio_threshold
        or row["text"].strip().endswith(":")
    )


def get_base_font_family(font_name):
    if "+" in font_name:
        font_name = font_name.split("+", 1)[-1]
    font_name = re.sub(r"(MT|PSMT|LT|BT|Std|Pro|Roman)", "", font_name)
    font_name = re.sub(r"[-_]?((Bold)?Italic|Bold|Regular)", "", font_name, flags=re.I)
    return font_name.strip("-_ ")


def regroup_into_paragraphs(df, space_threshold=5, font_tolerance=0.5):
    groups = []
    buffer = []
    prev = None
    group_id = 1
    same_line_threshold = 2

    for i, curr in df.iterrows():
        if prev is not None:
            same_line = abs(curr["top"] - prev["top"]) < same_line_threshold
            same_size = (curr["font_size"]/prev["font_size"] <= 1.2) and (curr["font_size"]/prev["font_size"] >= 0.8)
            
            same_base_font = (
                get_base_font_family(curr["font_name"]).lower()
                == get_base_font_family(prev["font_name"]).lower()
            )
            same_line = same_line and same_size
            if same_line:
                is_new_group = False
            else:
                is_new_group = (
                    curr["page"] != prev["page"]
                    or abs(curr["font_size"] - prev["font_size"]) > font_tolerance
                    or curr["space_above"] > space_threshold
                    or not same_base_font
                    or is_heading(prev)
                    or is_heading(curr)
                )

            if is_new_group:
                groups.append(merge_buffer(buffer, group_id))
                group_id += 1
                buffer = []

        buffer.append(curr)
        prev = curr

    if buffer:
        groups.append(merge_buffer(buffer, group_id))

    return pd.DataFrame(groups)


def merge_buffer(buffer, group_id):
    if not buffer:
        return None
    merged_text = " ".join(row["text"] for row in buffer)
    return {
        "group_id": group_id,
        "page": buffer[0]["page"],
        "text": merged_text,
        "font_size": round(buffer[0]["font_size"], 3),
        "font_name": buffer[0]["font_name"],
        "is_bold": buffer[0]["is_bold"],
        "x0": round(min(r["x0"] for r in buffer), 3),
        "x1": round(max(r["x1"] for r in buffer), 3),
        "top": round(buffer[0]["top"], 3),
        "bottom": round(buffer[-1]["bottom"], 3), 
        "width": round(max(r["x1"] for r in buffer) - min(r["x0"] for r in buffer), 3),
        "line_no": buffer[0]["line_no"],
        "space_above": round(buffer[0]["space_above"], 3),
        "space_below": round(buffer[-1]["space_below"], 3),
        "word_count": sum(r["word_count"] for r in buffer),
        "is_uppercase_ratio": round(
            sum(r["is_uppercase_ratio"] * len(r["text"]) for r in buffer)
            / max(sum(len(r["text"]) for r in buffer), 1),
            3,
        ),
        "is_landscape": buffer[0]["is_landscape"],
        "label": None,
    }


def add_text_features(df):
    """Add digit count and punctuation count features"""
    def count_digits(text):
        return sum(char.isdigit() for char in str(text))

    def count_punctuation(text):
        return sum(char in string.punctuation for char in str(text))

    df['num_count'] = df['text'].apply(count_digits)
    df['punctuation'] = df['text'].apply(count_punctuation)
    return df



def predict_labels(df, model_path, encoder_path):
    """Predict labels using trained XGBoost model"""
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    
    df['is_bold'] = df['is_bold'].astype(int)
    df['is_landscape'] = df['is_landscape'].astype(int)
    
   
    X = df[FEATURE_COLUMNS]
    y_pred_encoded = model.predict(X)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
    
    
    df['label'] = y_pred_labels
    
    return df



def generate_outline_json(df, pdf_name):
    """Generate JSON outline from labeled DataFrame"""

    titles = df[df['label'] == 'Title']['text'].dropna().str.strip()
    title_text = " | ".join(titles) if not titles.empty else pdf_name

    headings = df[df['label'].isin(['H1', 'H2', 'H3'])].copy()

    if headings.empty:
        return {
            "title": title_text,
            "outline": []
        }

    headings = headings.sort_values(['page', 'top']).reset_index(drop=True)

    outline = []
    for _, row in headings.iterrows():
        outline.append({
            "level": row['label'],
            "text": row['text'].strip(),
            "page": int(row['page'])
        })

    return {
        "title": title_text,
        "outline": outline
    }


def determine_heading_level(current_row, all_headings):
    """Determine the hierarchical level of a heading"""
    font_size = current_row['font_size']
    is_bold = current_row['is_bold']
    
   
    font_sizes = sorted(all_headings['font_size'].unique(), reverse=True)
    

    if font_size in font_sizes:
        level_num = font_sizes.index(font_size) + 1
        return f"h{min(level_num, 6)}"  
    
    return "h3"  



def process_single_pdf(pdf_path, output_folder, model_path, encoder_path):
    """Process a single PDF through the complete pipeline"""
    pdf_name = Path(pdf_path).stem
   
    df = extract_text_blocks(pdf_path)
    
    df = regroup_into_paragraphs(df)
    
    df = add_text_features(df)
    
    df = predict_labels(df, model_path, encoder_path)
    
    
    csv_path = os.path.join(output_folder, f"{pdf_name}.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
   

    outline_json = generate_outline_json(df, pdf_name)
    json_path = os.path.join(output_folder, f"{pdf_name}_outline.json")
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(outline_json, f, indent=2, ensure_ascii=False)
    
    return csv_path, json_path


def process_all_pdfs(input_folder, output_folder, model_path, encoder_path):
    """Process all PDFs in the input folder"""
    
    os.makedirs(output_folder, exist_ok=True)
    
    
    pdf_files = list(Path(input_folder).glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {input_folder}")
        return
    
    
    processed_files = []
    
    for pdf_path in pdf_files:
        try:
            csv_path, json_path = process_single_pdf(
                str(pdf_path), output_folder, model_path, encoder_path
            )
            processed_files.append({
                'pdf': str(pdf_path),
                'csv': csv_path,
                'json': json_path
            })
        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {str(e)}")
            continue
    
    
    
    summary_path = os.path.join(output_folder, "processing_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_processed': len(processed_files),
            'files': processed_files
        }, f, indent=2, ensure_ascii=False)
        print("✅ Processing done")
    



if __name__ == "__main__":
    
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        exit(1)
    
    if not os.path.exists(ENCODER_PATH):
        print(f"❌ Encoder file not found: {ENCODER_PATH}")
        exit(1)
    

    process_all_pdfs(INPUT_FOLDER, OUTPUT_FOLDER, MODEL_PATH, ENCODER_PATH)
