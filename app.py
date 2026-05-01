import streamlit as st
import pandas as pd
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis
from ultralytics import YOLO
import io
from datetime import datetime
import os
import torch
import traceback
import json
import subprocess
import sys
import tempfile
import shutil
import yaml
import time
import re
import threading
import glob

from mysql_handler import (
    create_connection, create_table, insert_data,
    insert_defect_info, fetch_all_entries, fetch_all_defect_info
)

# ---------------- Configuration ----------------
DISPLAY_SIZE = (600, 600)
MODEL_SIZE = 1280
ZOOM_SIZE = (250, 250)
CONFIDENCE_THRESHOLD = 0.5
MODELS_DIR = "models"

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# ---------------- Training Log Cleaner ----------------
def clean_training_log(text):
    """Strip all terminal escape codes and control characters for clean readable output."""
    # Remove ANSI escape sequences (colors, cursor moves, etc.)
    text = re.sub(r'\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)
    # Remove leftover [K (erase-to-end-of-line remnants)
    text = re.sub(r'\[K', '', text)
    # Handle carriage returns: keep only the last segment on each line
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        parts = line.split('\r')
        clean_lines.append(parts[-1].rstrip())
    # Remove consecutive duplicate lines (progress bar spam)
    deduped = []
    prev = None
    for line in clean_lines:
        stripped = line.strip()
        if stripped and stripped == prev:
            continue
        deduped.append(line)
        prev = stripped
    return '\n'.join(deduped)

def parse_training_line(line):
    """Try to extract epoch/loss info from a training log line."""
    # Match lines like: "  9/50   4.36G   3.638   3.629   2.043   11   1280: 100%"
    m = re.match(r'\s*(\d+)/(\d+)\s+([\d.]+G)?\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)', line)
    if m:
        return {
            'epoch': int(m.group(1)),
            'total': int(m.group(2)),
            'box_loss': m.group(4),
            'cls_loss': m.group(5),
            'dfl_loss': m.group(6),
        }
    # Match validation result: "all   39   120   0.455   0.363   0.287   0.179"
    m2 = re.match(r'\s*all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
    if m2:
        return {
            'val': True,
            'images': m2.group(1),
            'instances': m2.group(2),
            'precision': m2.group(3),
            'recall': m2.group(4),
            'mAP50': m2.group(5),
            'mAP50_95': m2.group(6),
        }
    return None


# ---------------- Model Management ----------------
def get_available_models():
    models = []
    if os.path.exists('best.pt'):
        models.append(('best.pt', 'Legacy Model (best.pt)', os.path.getmtime('best.pt')))
    if os.path.exists(MODELS_DIR):
        for file in os.listdir(MODELS_DIR):
            if file.endswith('.pt'):
                full_path = os.path.join(MODELS_DIR, file)
                models.append((full_path, file, os.path.getmtime(full_path)))
    models.sort(key=lambda x: x[2], reverse=True)
    return models

def get_next_model_version():
    existing_models = get_available_models()
    max_version = 0
    for path, name, _ in existing_models:
        if 'training_' in name:
            try:
                version = int(name.replace('training_', '').replace('.pt', '').replace('_last', ''))
                max_version = max(max_version, version)
            except:
                pass
    return max_version + 1

def clear_training_files():
    """Delete all temp training directories and leftover model backups."""
    cleared = []
    # Temp dirs created by tempfile with our prefix
    for d in glob.glob(os.path.join(tempfile.gettempdir(), 'st_train_*')):
        try:
            shutil.rmtree(d)
            cleared.append(d)
        except Exception as e:
            pass
    # Temp model backups in current directory
    for f in glob.glob('temp_model_*.pt'):
        try:
            os.remove(f)
            cleared.append(f)
        except:
            pass
    # YOLO runs directory
    if os.path.exists('runs'):
        try:
            shutil.rmtree('runs')
            cleared.append('runs/')
        except:
            pass
    # Training log files
    for f in glob.glob('training_log_*.txt'):
        try:
            os.remove(f)
            cleared.append(f)
        except:
            pass
    return cleared


# ---------------- Database CRUD helpers ----------------
def delete_defect_data(record_id):
    try:
        conn = create_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM defect_data WHERE id = %s", (record_id,))
            conn.commit()
            cursor.close()
            conn.close()
            return True
    except Exception as e:
        st.error(f"Delete error: {e}")
        traceback.print_exc()
    return False

def delete_defect_info_record(record_id):
    try:
        conn = create_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM defect_info WHERE id = %s", (record_id,))
            conn.commit()
            cursor.close()
            conn.close()
            return True
    except Exception as e:
        st.error(f"Delete error: {e}")
        traceback.print_exc()
    return False

def update_defect_data(record_id, fields: dict):
    try:
        conn = create_connection()
        if conn:
            cursor = conn.cursor()
            set_clause = ", ".join([f"{k} = %s" for k in fields.keys()])
            values = list(fields.values()) + [record_id]
            cursor.execute(f"UPDATE defect_data SET {set_clause} WHERE id = %s", values)
            conn.commit()
            cursor.close()
            conn.close()
            return True
    except Exception as e:
        st.error(f"Update error: {e}")
        traceback.print_exc()
    return False

def update_defect_info_record(record_id, fields: dict):
    try:
        conn = create_connection()
        if conn:
            cursor = conn.cursor()
            set_clause = ", ".join([f"{k} = %s" for k in fields.keys()])
            values = list(fields.values()) + [record_id]
            cursor.execute(f"UPDATE defect_info SET {set_clause} WHERE id = %s", values)
            conn.commit()
            cursor.close()
            conn.close()
            return True
    except Exception as e:
        st.error(f"Update error: {e}")
        traceback.print_exc()
    return False


st.set_page_config(page_title="X-ray Defect Detection", page_icon="🔍", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #2196F3; text-align: center; padding: 20px 0; }
    .sub-header { font-size: 1.8rem; font-weight: 600; color: #FF6B35; margin: 20px 0; }
    .info-box { background-color: #1a1a2e; color: #fff; padding: 20px; border-radius: 10px; border-left: 5px solid #2196F3; margin: 15px 0; }
    .success-box { background-color: #1b5e20; color: #fff; padding: 20px; border-radius: 10px; border-left: 5px solid #4caf50; margin: 15px 0; }
    .warning-box { background-color: #4a2c2a; color: #fff; padding: 20px; border-radius: 10px; border-left: 5px solid #ff9800; margin: 15px 0; }
    .defect-info { background-color: #2d2d2d; color: #fff; padding: 15px; border-radius: 8px; margin: 10px 0; }
    .train-log { font-family: monospace; font-size: 0.85rem; background: #0e1117; color: #e0e0e0;
                 padding: 12px; border-radius: 6px; max-height: 400px; overflow-y: auto;
                 white-space: pre-wrap; word-break: break-all; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# ---------------- Utility functions ----------------
def standardize_image_for_display(image, target_size=DISPLAY_SIZE):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    background = Image.new('RGB', target_size, (255, 255, 255))
    offset = ((target_size[0] - image.size[0]) // 2, (target_size[1] - image.size[1]) // 2)
    background.paste(image, offset)
    return background

def enhance_roi_display(roi):
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    border = 15
    bordered = cv2.copyMakeBorder(rgb, border, border, border, border,
                                  cv2.BORDER_CONSTANT, value=[100, 150, 255])
    return bordered

def create_zoom_view(image, bbox, zoom_size=ZOOM_SIZE):
    x1, y1, x2, y2 = map(int, bbox)
    pad = 10
    y1_pad = max(0, y1 - pad)
    y2_pad = min(image.shape[0], y2 + pad)
    x1_pad = max(0, x1 - pad)
    x2_pad = min(image.shape[1], x2 + pad)
    roi = image[y1_pad:y2_pad, x1_pad:x2_pad]
    if roi.size == 0:
        return np.ones((zoom_size[1], zoom_size[0], 3), dtype=np.uint8) * 255
    roi_pil = Image.fromarray(roi)
    roi_pil = roi_pil.resize(zoom_size, Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Sharpness(roi_pil)
    roi_pil = enhancer.enhance(3.0)
    enhancer = ImageEnhance.Contrast(roi_pil)
    roi_pil = enhancer.enhance(1.5)
    roi_array = np.array(roi_pil)
    gaussian = cv2.GaussianBlur(roi_array, (0, 0), 2.0)
    roi_array = cv2.addWeighted(roi_array, 2.5, gaussian, -1.5, 0)
    return np.clip(roi_array, 0, 255).astype(np.uint8)

def convert_to_bytes(img):
    if isinstance(img, np.ndarray):
        is_success, buffer = cv2.imencode(".png", img)
        if is_success:
            return buffer.tobytes()
    elif isinstance(img, Image.Image):
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    return None

def apply_clahe(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def apply_histogram_equalization(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return cv2.equalizeHist(gray)

def apply_contrast_enhancement(image, alpha=1.5):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    clahe_img = apply_clahe(gray)
    return cv2.convertScaleAbs(clahe_img, alpha=alpha, beta=0)

def apply_sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def apply_denoising(image):
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

def apply_gaussian_blur(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_edge_detection(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)

def extract_all_features(roi):
    if len(np.array(roi).shape) == 3:
        gray_roi = cv2.cvtColor(np.array(roi), cv2.COLOR_BGR2GRAY)
    else:
        gray_roi = np.array(roi)
    features = {}
    try:
        glcm = graycomatrix(gray_roi, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                           levels=256, symmetric=True, normed=True)
        features['Contrast'] = float(graycoprops(glcm, 'contrast').mean())
        features['Correlation'] = float(graycoprops(glcm, 'correlation').mean())
        features['Energy'] = float(graycoprops(glcm, 'energy').mean())
        features['Homogeneity'] = float(graycoprops(glcm, 'homogeneity').mean())
        features['Entropy'] = float(shannon_entropy(gray_roi))
        features['Skewness'] = float(skew(gray_roi.flatten()))
        features['Kurtosis'] = float(kurtosis(gray_roi.flatten()))
        features['Mean_Intensity'] = float(np.mean(gray_roi))
        features['Std_Intensity'] = float(np.std(gray_roi))
        features['Min_Intensity'] = float(np.min(gray_roi))
        features['Max_Intensity'] = float(np.max(gray_roi))
        h, w = gray_roi.shape
        features['Width'] = w
        features['Height'] = h
        features['Area'] = w * h
        features['Aspect_Ratio'] = float(w / h) if h > 0 else 0
    except Exception:
        h, w = gray_roi.shape
        features = {'Width': w, 'Height': h, 'Area': w*h, 'Mean_Intensity': float(np.mean(gray_roi))}
    return features

# ---------------- Model loading ----------------
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Error: {e}")
        traceback.print_exc()
        return None

def predict(image, model):
    try:
        h, w = image.shape[:2]
        if h != MODEL_SIZE or w != MODEL_SIZE:
            image_resized = cv2.resize(image, (MODEL_SIZE, MODEL_SIZE))
        else:
            image_resized = image
        results = model(image_resized)
        return results, image_resized
    except Exception as e:
        st.error(f"Prediction error: {e}")
        traceback.print_exc()
        return None, None

def draw_detections(image, results, model):
    segmented_images = []
    defects_info = []
    zoom_views = []
    image_with_defects = image.copy()
    DEFECT_CLASSES = ['porosity', 'lof', 'linear_pore', 'LOP', 'linear pore']
    defect_count = 0
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        for obj in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = obj[:6]
            label = model.names[int(cls)]
            if any(d.lower() in label.lower() for d in DEFECT_CLASSES):
                if conf > CONFIDENCE_THRESHOLD:
                    defect_count += 1
                    cv2.rectangle(image_with_defects, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    cv2.putText(image_with_defects, f'{label} {conf:.2f}',
                               (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    roi = image[max(0, int(y1)):min(image.shape[0], int(y2)),
                               max(0, int(x1)):min(image.shape[1], int(x2))]
                    if roi.size > 0:
                        segmented_images.append(roi)
                        zoom_views.append(create_zoom_view(image, (x1, y1, x2, y2)))
                        defects_info.append({'type': label, 'confidence': float(conf),
                                            'bbox': (float(x1), float(y1), float(x2), float(y2))})
    return image_with_defects, segmented_images, defects_info, zoom_views, defect_count

# ---------------- Session-state helpers ----------------
def clear_analysis_state():
    keys = ["analysis_done", "processed_image", "results", "img_defects", "rois",
            "defects_info", "zoom_views", "defect_count", "features_df", "all_features", "last_upload_name"]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]

def save_analysis_to_session(processed_image, results, img_defects, rois, defects_info,
                              zoom_views, defect_count, df, all_features, upload_name):
    st.session_state["analysis_done"] = True
    st.session_state["processed_image"] = processed_image
    st.session_state["results"] = results
    st.session_state["img_defects"] = img_defects
    st.session_state["rois"] = rois
    st.session_state["defects_info"] = defects_info
    st.session_state["zoom_views"] = zoom_views
    st.session_state["defect_count"] = defect_count
    st.session_state["features_df"] = df
    st.session_state["all_features"] = all_features
    st.session_state["last_upload_name"] = upload_name

# ---------------- Training session state init ----------------
for _k, _v in [
    ('training_active', False),
    ('training_logs', ''),
    ('training_process', None),
    ('training_result', None),
    ('training_tmp_dir', None),
    ('training_safe_model_copy', None),
    ('training_candidate_best', None),
    ('training_candidate_last', None),
    ('training_log_file', None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------- UI ----------------
st.markdown('<p class="main-header">🔍 X-ray Defect Detection System</p>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/xray.png", width=80)
    st.markdown("### Navigation")
    
    # Training status indicator
    if st.session_state.training_active:
        st.info("⏳ Training in progress...")
        st.caption("Go to Training tab to monitor")
    elif st.session_state.training_result:
        result_status = st.session_state.training_result.get('status')
        if result_status == 'success':
            st.success("✅ Training completed!")
            st.caption("Go to Training tab to use model")
        elif result_status == 'failed':
            st.error("❌ Training failed")
            st.caption("Check Training tab for details")
        elif result_status == 'partial':
            st.warning("⚠️ Training completed (last.pt)")
            st.caption("Go to Training tab to use model")
        elif result_status == 'stopped':
            st.warning("⚠️ Training was stopped")
            st.caption("Go to Training tab to restart")
    
    page = st.radio("", ["🔍 Detection", "🎨 Processing", "📚 Training", "💾 Database", "ℹ️ Help"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 🤖 Model Selection")
    available_models = get_available_models()
    if available_models:
        model_options = [f"{name} ({datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')})"
                        for path, name, mtime in available_models]
        model_paths = [path for path, name, mtime in available_models]
        if 'selected_model_idx' not in st.session_state:
            st.session_state.selected_model_idx = 0
        # Safety check: ensure index is valid
        if st.session_state.selected_model_idx >= len(available_models):
            st.session_state.selected_model_idx = 0
        selected_idx = st.selectbox("Choose model:", range(len(model_options)),
                                    format_func=lambda i: model_options[i],
                                    index=st.session_state.selected_model_idx,
                                    key="model_selector")
        st.session_state.selected_model_idx = selected_idx
        st.session_state.selected_model_path = model_paths[selected_idx]
        st.success(f"✅ {len(available_models)} model(s) available")
        model_size = os.path.getsize(model_paths[selected_idx]) / (1024 * 1024)
        st.caption(f"Size: {model_size:.1f} MB")
    else:
        st.warning("⚠️ No models found")
        st.session_state.selected_model_path = None

# ══════════════════════════════════════════════════════
# DETECTION PAGE
# ══════════════════════════════════════════════════════
if page == "🔍 Detection":
    st.markdown('<p class="sub-header">Defect Detection</p>', unsafe_allow_html=True)
    if not st.session_state.get('selected_model_path'):
        st.error("❌ No model selected! Upload or train a model first.")
        st.stop()
    selected_model_path = st.session_state.selected_model_path
    model = load_model(selected_model_path)
    if not model:
        st.error(f"❌ Failed to load model: {selected_model_path}")
        st.stop()
    st.info(f"🤖 Using model: **{os.path.basename(selected_model_path)}**")
    st.markdown('<div class="info-box">📤 Upload X-ray image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        if st.session_state.get("last_upload_name") != uploaded_file.name:
            clear_analysis_state()
            st.session_state["last_upload_name"] = uploaded_file.name
    if uploaded_file:
        original_image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### 📤 Uploaded")
            st.image(standardize_image_for_display(original_image), use_column_width=True)
        with col2:
            st.markdown("#### 📊 Info")
            st.write(f"**Size:** {original_image.size[0]} × {original_image.size[1]} px")
        if st.button("🔎 Analyze", type="primary"):
            with st.spinner('Analyzing at 1280×1280...'):
                image_array = np.array(original_image.convert('RGB'))
                results, processed_image = predict(image_array, model)
                if not results:
                    st.error("❌ Analysis failed")
                else:
                    img_defects, rois, defects_info, zoom_views, defect_count = draw_detections(processed_image, results, model)
                    all_features = []
                    for idx, roi in enumerate(rois):
                        features = extract_all_features(roi)
                        features['ID'] = f"#{idx+1}"
                        features['Type'] = defects_info[idx]['type']
                        features['Conf'] = f"{defects_info[idx]['confidence']:.1%}"
                        all_features.append(features)
                    df = pd.DataFrame(all_features)
                    save_analysis_to_session(processed_image, results, img_defects, rois, defects_info,
                                            zoom_views, defect_count, df, all_features, uploaded_file.name)
                    st.success(f"✅ Found {defect_count} defect(s)")
        if st.session_state.get("analysis_done"):
            img_defects = st.session_state["img_defects"]
            rois = st.session_state["rois"]
            defects_info = st.session_state["defects_info"]
            zoom_views = st.session_state["zoom_views"]
            defect_count = st.session_state["defect_count"]
            df = st.session_state["features_df"]
            st.markdown("---")
            st.markdown("### 🎯 Detection Results")
            st.image(standardize_image_for_display(img_defects), use_column_width=True)
            st.download_button("📥 Download", convert_to_bytes(img_defects),
                               f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            if defect_count > 0:
                st.markdown("---")
                st.markdown(f"### 🔬 Analysis ({defect_count} defects)")
                for idx, (roi, info, zoom) in enumerate(zip(rois, defects_info, zoom_views)):
                    st.markdown(f"#### Defect #{idx + 1}: {info['type']}")
                    col2, col3 = st.columns([1.2, 1.2])
                    with col2:
                        st.markdown("**Zoom**")
                        st.image(zoom)
                    with col3:
                        st.markdown('<div class="defect-info">', unsafe_allow_html=True)
                        st.markdown(f"**Type:** {info['type']}")
                        st.markdown(f"**Confidence:** {info['confidence']:.1%}")
                        col_a, col_b = st.columns(2)
                        with col_b:
                            st.download_button("📥 Zoom", convert_to_bytes(zoom),
                                               f"d{idx+1}_zoom.png", key=f"z{idx}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("---")
                st.markdown("### 📊 Features")
                st.dataframe(df, use_container_width=True)
                st.download_button("📥 CSV", df.to_csv(index=False),
                                   f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                st.markdown("---")
                st.markdown("### 💾 Save to Database")
                with st.expander("💾 Click to save", expanded=False):
                    with st.form("db_save"):
                        c1, c2 = st.columns(2)
                        with c1:
                            defect_no = st.text_input("Defect No*", placeholder="DEF-2025-001", key="form_defect_no")
                            satellite = st.text_input("Satellite*", placeholder="SAT-XR-100", key="form_satellite")
                            component = st.text_input("Component*", placeholder="Fuel Tank", key="form_component")
                        with c2:
                            comp_id = st.text_input("Component ID", value=uploaded_file.name, key="form_comp_id")
                            date = st.date_input("Date", key="form_date")
                            status = st.radio("Status", ["Accept", "Reject"], horizontal=True, key="form_status")
                        remarks = st.text_area("Remarks", key="form_remarks")
                        st.write("**Defects:**", ", ".join([f"{d['type']} ({d['confidence']:.0%})" for d in defects_info]))
                        if st.form_submit_button("💾 Save", type="primary"):
                            if defect_no and satellite and component:
                                try:
                                    defect_list = ", ".join([d['type'] for d in defects_info])
                                    features_json = df.to_json(orient='records')
                                    try:
                                        import mysql_handler as mh
                                        st.write("DB host:", mh.DB_CONFIG.get('host'),
                                                 "DB:", mh.DB_CONFIG.get('database'),
                                                 "User:", mh.DB_CONFIG.get('user'))
                                    except Exception as e:
                                        st.write("mysql_handler not importable:", e)
                                    s1 = insert_data(defect_no, satellite, component, comp_id, defect_list, date)
                                    s2 = insert_defect_info(defect_no, defect_list, features_json, remarks, status)
                                    st.write("insert results -> s1:", s1, " s2:", s2)
                                    try:
                                        rows = fetch_all_entries()
                                        rows_info = fetch_all_defect_info()
                                        st.write("Latest defect_data rows (top 5):")
                                        st.write(rows[:5])
                                        st.write("Latest defect_info rows (top 5):")
                                        st.write(rows_info[:5])
                                    except Exception as e:
                                        st.error(f"fetch_all_* threw: {e}")
                                        traceback.print_exc()
                                    if s1 and s2:
                                        st.success("✅ Saved!")
                                        st.balloons()
                                    else:
                                        st.error("Save failed — check errors above / console logs")
                                except Exception as e:
                                    st.error(f"Save exception: {e}")
                                    traceback.print_exc()
                            else:
                                st.error("Fill required (*) fields")
        else:
            st.info("Click Analyze to detect defects")
    else:
        st.info("Upload an X-ray image to analyze")

# ══════════════════════════════════════════════════════
# PROCESSING PAGE
# ══════════════════════════════════════════════════════
elif page == "🎨 Processing":
    st.markdown('<p class="sub-header">Image Processing</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_cv = np.array(image.convert('RGB'))
        st.markdown("#### Original")
        st.image(standardize_image_for_display(image))
        st.markdown("---")
        select_all = st.checkbox("✅ Select All")
        c1, c2 = st.columns(2)
        with c1:
            h = st.checkbox("Hist Eq", value=select_all)
            c = st.checkbox("CLAHE", value=True if not select_all else select_all)
            ct = st.checkbox("Contrast", value=select_all)
            s = st.checkbox("Sharp", value=select_all)
        with c2:
            d = st.checkbox("Denoise", value=select_all)
            b = st.checkbox("Blur", value=select_all)
            e = st.checkbox("Edges", value=select_all)
        if st.button("Apply", type="primary"):
            processed = []
            titles = []
            gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
            processed.append(gray); titles.append("Gray")
            if h: processed.append(apply_histogram_equalization(gray)); titles.append("HistEq")
            if c: processed.append(apply_clahe(gray)); titles.append("CLAHE")
            if ct: processed.append(apply_contrast_enhancement(gray)); titles.append("Contrast")
            if s: processed.append(apply_sharpening(apply_clahe(gray) if c else gray)); titles.append("Sharp")
            if d:
                dn = apply_denoising(gray)
                if len(dn.shape) == 3:
                    dn = cv2.cvtColor(dn, cv2.COLOR_RGB2GRAY)
                processed.append(dn); titles.append("Denoise")
            if b: processed.append(apply_gaussian_blur(gray)); titles.append("Blur")
            if e: processed.append(apply_edge_detection(gray)); titles.append("Edges")
            st.markdown("---")
            for i in range(0, len(processed), 3):
                cols = st.columns(3)
                for j in range(3):
                    idx = i + j
                    if idx < len(processed):
                        with cols[j]:
                            st.image(standardize_image_for_display(Image.fromarray(processed[idx]), (300, 300)),
                                    caption=titles[idx])
                            st.download_button(f"📥", convert_to_bytes(processed[idx]),
                                             f"{titles[idx]}.png", key=f"p{idx}")

# ══════════════════════════════════════════════════════
# TRAINING PAGE
# ══════════════════════════════════════════════════════
elif page == "📚 Training":
    st.markdown('<p class="sub-header">Training</p>', unsafe_allow_html=True)
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        st.success(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("⚠️ No GPU — training will be slow on CPU")

    tab1, tab2, tab3 = st.tabs(["📤 Upload Model", "🎓 Train", "🗂️ Manage Models"])

    # ── Tab 1: Upload Model ──────────────────────────────────────────────────
    with tab1:
        st.markdown("### Upload Pre-trained Model")
        model_file = st.file_uploader("Choose .pt model file", type=["pt"], key="mup")
        if model_file:
            col1, col2 = st.columns(2)
            with col1:
                model_name = st.text_input("Model name (optional)", placeholder="my_custom_model",
                                           key="model_name_input")
            if st.button("💾 Save Model", type="primary"):
                if model_name:
                    filename = f"{model_name}.pt"
                else:
                    filename = model_file.name if model_file.name.endswith('.pt') \
                               else f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                save_path = os.path.join(MODELS_DIR, filename)
                if os.path.exists(save_path):
                    st.warning(f"⚠️ {filename} already exists. Overwriting...")
                with open(save_path, 'wb') as f:
                    f.write(model_file.read())
                st.success(f"✅ Model saved as: {filename}")
                st.balloons()
                time.sleep(1)
                st.rerun()

    # ── Tab 2: Train ─────────────────────────────────────────────────────────
    with tab2:

        # ---- LIVE TRAINING VIEW (shown when training is running) ----
        if st.session_state.training_active:
            st.markdown("### ⏳ Training In Progress")
            
            # Read logs from file
            log_file = st.session_state.training_log_file
            current_logs = ""
            if log_file and os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        current_logs = f.read()
                except Exception as e:
                    current_logs = f"Error reading log file: {e}"
            
            progress_placeholder = st.empty()
            log_placeholder = st.empty()
            
            stop_col, _ = st.columns([1, 3])
            with stop_col:
                if st.button("🛑 Stop Training", type="secondary"):
                    # Check if result already set (training completed in background)
                    if st.session_state.training_result:
                        st.info("✅ Training already completed! Showing results...")
                        st.session_state.training_active = False
                        st.session_state.training_process = None
                        time.sleep(0.3)
                        st.rerun()
                    else:
                        # Actually stop the running process
                        proc = st.session_state.training_process
                        is_running = False
                        if proc and proc.poll() is None:
                            # Process is still running
                            is_running = True
                            try:
                                proc.kill()
                                proc.wait(timeout=5)
                            except Exception:
                                pass
                        
                        if is_running:
                            # Only set stopped status if we actually killed a running process
                            st.session_state.training_active = False
                            st.session_state.training_result = {'status': 'stopped'}
                            # Clean up temp model backup
                            smc = st.session_state.training_safe_model_copy
                            if smc and os.path.exists(smc) and not smc.startswith('yolov8'):
                                try: os.remove(smc)
                                except: pass
                            st.warning("⚠️ Training stopped by user.")
                        else:
                            # Process already finished, just trigger refresh to show results
                            st.info("⏳ Training completed, loading results...")
                            st.session_state.training_active = False
                        
                        time.sleep(0.3)
                        st.rerun()

            # Parse and show clean live progress
            if not current_logs:
                current_logs = "⏳ Starting training process, please wait...\n\nIf you see this for more than 30 seconds, check your training_handler.py file."
            
            log_lines = current_logs.split('\n')

            # Find last epoch line for headline metric
            last_epoch_info = None
            for line in reversed(log_lines):
                info = parse_training_line(line)
                if info and 'epoch' in info:
                    last_epoch_info = info
                    break

            if last_epoch_info:
                ep = last_epoch_info['epoch']
                total = last_epoch_info['total']
                progress_val = ep / total
                progress_placeholder.progress(progress_val,
                    text=f"Epoch {ep}/{total}  |  box_loss: {last_epoch_info['box_loss']}  "
                         f"cls_loss: {last_epoch_info['cls_loss']}  dfl_loss: {last_epoch_info['dfl_loss']}")
            else:
                progress_placeholder.info("⏳ Initialising model and dataset…")

            # Show last 60 clean lines
            clean_tail = '\n'.join(log_lines[-60:])
            log_placeholder.code(clean_tail, language="")

            # Check if training just finished (background thread completed)
            if not st.session_state.training_active:
                # Training finished! Clear process reference and show results
                st.session_state.training_process = None
                time.sleep(0.5)
                st.rerun()
            
            # Poll for updates (only if still active)
            time.sleep(1.5)
            st.rerun()

        # ---- TRAINING RESULT VIEW ----
        elif st.session_state.training_result:
            result = st.session_state.training_result

            # Debug expander (collapsible)
            with st.expander("🔧 Debug Info", expanded=False):
                st.json(result)
                st.write("**Session State:**")
                st.write(f"- training_active: {st.session_state.training_active}")
                st.write(f"- training_process: {st.session_state.training_process}")
                st.write(f"- Result status: {result.get('status')}")

            if result.get('status') == 'stopped':
                st.warning("⚠️ Training was stopped by user.")
                st.info("💡 **Tip:** If training was actually complete, check the models/ directory - the model might have been saved anyway!")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📂 Check Models", type="primary"):
                        st.rerun()
                with col2:
                    if st.button("🔄 Start New Training", type="secondary"):
                        st.session_state.training_result = None
                        st.session_state.training_logs = ''
                        st.session_state.training_process = None
                        st.session_state.training_log_file = None
                        st.rerun()

            elif result.get('status') == 'success':
                st.success(f"✅ Training complete! New model saved as: **{result['model_name']}**")
                st.balloons()
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Use this model for detection", type="primary"):
                        avail = get_available_models()
                        for idx, (path, name, mtime) in enumerate(avail):
                            if path == result['model_path']:
                                st.session_state.selected_model_idx = idx
                                st.session_state.selected_model_path = result['model_path']
                                st.success(f"✅ Now using {result['model_name']}")
                                time.sleep(1)
                                st.rerun()
                                break
                with col2:
                    if st.button("🔄 Start New Training", type="secondary"):
                        st.session_state.training_result = None
                        st.session_state.training_logs = ''
                        st.session_state.training_process = None
                        st.session_state.training_log_file = None
                        st.rerun()

            elif result.get('status') == 'partial':
                st.warning(f"⚠️ No best.pt found — saved last checkpoint as: **{result['model_name']}**")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Use this model anyway"):
                        avail = get_available_models()
                        for idx, (path, name, mtime) in enumerate(avail):
                            if path == result['model_path']:
                                st.session_state.selected_model_idx = idx
                                st.session_state.selected_model_path = result['model_path']
                                st.rerun()
                                break
                with col2:
                    if st.button("🔄 Start New Training", type="secondary"):
                        st.session_state.training_result = None
                        st.session_state.training_logs = ''
                        st.session_state.training_process = None
                        st.session_state.training_log_file = None
                        st.rerun()

            elif result.get('status') == 'failed':
                st.error("❌ Training failed.")
                st.write(f"Return code: {result.get('rc')}")
                log_file = st.session_state.training_log_file
                if log_file and os.path.exists(log_file):
                    with st.expander("Show full training log"):
                        try:
                            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                                st.code(f.read(), language="")
                        except Exception as e:
                            st.error(f"Could not read log file: {e}")
                if st.button("🔄 Start New Training Session"):
                    st.session_state.training_result = None
                    st.session_state.training_logs = ''
                    st.session_state.training_process = None
                    st.session_state.training_log_file = None
                    st.rerun()

        # ---- CONFIGURATION & START BUTTON ----
        else:
            st.info("Configure parameters below, then press Start Training.")
            
            # ---- Training Mode Selection ----
            st.markdown("### 🎯 Training Mode")
            training_modes = []
            
            # Check if best.pt exists
            if os.path.exists('best.pt'):
                training_modes.append("Use best.pt (legacy model)")
            
            # Check if a model is currently selected
            if st.session_state.get('selected_model_path') and \
                    os.path.exists(st.session_state.selected_model_path):
                selected_name = os.path.basename(st.session_state.selected_model_path)
                training_modes.append(f"Transfer learning from {selected_name}")
            
            # Always allow training from scratch
            training_modes.append("Train from scratch (new YOLO model)")
            
            training_mode = st.radio(
                "Select training approach:",
                training_modes,
                help="Choose whether to continue training an existing model or start fresh"
            )
            
            # YOLO model selection for scratch training
            yolo_model = None
            if "scratch" in training_mode.lower():
                st.markdown("**Select YOLO Model:**")
                yolo_models = {
                    "YOLOv8 Nano (fastest, least accurate)": "yolov8n.pt",
                    "YOLOv8 Small (balanced)": "yolov8s.pt",
                    "YOLOv8 Medium (good accuracy)": "yolov8m.pt",
                    "YOLOv8 Large (high accuracy)": "yolov8l.pt",
                    "YOLOv8 XLarge (best accuracy, slowest)": "yolov8x.pt",
                }
                selected_yolo = st.selectbox(
                    "Choose base model:",
                    list(yolo_models.keys()),
                    index=1,  # Default to Small
                    help="Smaller models train faster but may be less accurate. Larger models are more accurate but slower."
                )
                yolo_model = yolo_models[selected_yolo]
                st.info(f"📦 Will download and use: **{yolo_model}**")
            
            st.markdown("---")
            st.markdown("### ⚙️ Training Parameters")
            c1, c2 = st.columns(2)
            with c1:
                images = st.file_uploader("Images (train set)", type=["jpg", "png"],
                                          accept_multiple_files=True, key="train_images")
                labels = st.file_uploader("Labels (.txt YOLO format)", type=["txt"],
                                          accept_multiple_files=True, key="train_labels")
            with c2:
                epochs = st.number_input("Epochs", min_value=1, value=50, step=1)
                batch = st.number_input("Batch size", min_value=1, value=4, step=1)
                imgsz = st.number_input("Image size (imgsz)", min_value=128, value=1280, step=1)
                lr0 = st.number_input("Initial LR (lr0)", value=0.001, format="%.6f")
                lrf = st.number_input("Final LR factor (lrf)", value=0.01, format="%.6f")
                optimizer = st.selectbox("Optimizer", ["Adam", "SGD"], index=0)
                weight_decay = st.number_input("Weight decay", value=0.0005, format="%.6f")
                st.markdown("**Augmentation**")
                use_mosaic = st.slider("Mosaic (0.0 – 1.0)", 0.0, 1.0, 0.5)
                degrees = st.slider("Rotation degrees", 0.0, 180.0, 10.0)
                translate = st.slider("Translate", 0.0, 1.0, 0.1)
                scale = st.slider("Scale", 0.1, 2.0, 0.5)
                fliplr = st.slider("Horizontal flip prob", 0.0, 1.0, 0.5)
                use_clahe = st.checkbox("Apply CLAHE preprocessing to images", value=False)
                st.markdown("**Data Validation**")
                min_images = st.number_input(
                    "Minimum matched image-label pairs to allow training",
                    min_value=1, value=10, step=1,
                    help="Training is blocked if matched pairs fall below this number.")

            # Validation summary
            if images and labels:
                img_stems = set(os.path.splitext(f.name)[0] for f in images)
                lbl_stems = set(os.path.splitext(f.name)[0] for f in labels)
                matched = len(img_stems & lbl_stems)
                unmatched_imgs = img_stems - lbl_stems
                unmatched_lbls = lbl_stems - img_stems
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1: st.metric("✅ Matched pairs", matched)
                with col_m2: st.metric("🖼️ Images", len(images))
                with col_m3: st.metric("🏷️ Labels", len(labels))
                if unmatched_imgs:
                    st.warning(f"⚠️ {len(unmatched_imgs)} image(s) without a label: "
                               f"{', '.join(list(unmatched_imgs)[:5])}{'…' if len(unmatched_imgs)>5 else ''}")
                if unmatched_lbls:
                    st.warning(f"⚠️ {len(unmatched_lbls)} label(s) without an image: "
                               f"{', '.join(list(unmatched_lbls)[:5])}{'…' if len(unmatched_lbls)>5 else ''}")
                if matched < min_images:
                    st.error(f"❌ Only **{matched}** matched pair(s) — minimum is **{min_images}**. "
                             f"Upload {min_images - matched} more pair(s) or lower the threshold.")
                else:
                    st.success(f"✅ {matched} matched pairs — meets minimum of {min_images}. Ready!")
            else:
                st.caption("Upload images and labels above to see validation stats.")

            # ---- START TRAINING ----
            if st.button("▶️ Start Training", type="primary"):
                if not images or not labels:
                    st.error("Please upload both images and label files.")
                else:
                    img_stems = set(os.path.splitext(f.name)[0] for f in images)
                    lbl_stems = set(os.path.splitext(f.name)[0] for f in labels)
                    matched_count = len(img_stems & lbl_stems)
                    if matched_count < min_images:
                        st.error(
                            f"❌ Training blocked! Only **{matched_count}** matched pair(s) found, "
                            f"but minimum required is **{min_images}**. "
                            f"Please upload {min_images - matched_count} more matched pair(s)."
                        )
                    else:
                        # Set up temp directory
                        tmp_dir = tempfile.mkdtemp(prefix="st_train_")
                        images_dir = os.path.join(tmp_dir, "images")
                        labels_dir = os.path.join(tmp_dir, "labels")
                        os.makedirs(images_dir, exist_ok=True)
                        os.makedirs(labels_dir, exist_ok=True)
                        for img in images:
                            with open(os.path.join(images_dir, img.name), "wb") as f:
                                f.write(img.read())
                        for lbl in labels:
                            with open(os.path.join(labels_dir, lbl.name), "wb") as f:
                                f.write(lbl.read())
                        data_config = {
                            'path': tmp_dir, 'train': 'images', 'val': 'images',
                            'nc': 3, 'names': ['LOP', 'linear pore', 'porosity']
                        }
                        with open(os.path.join(tmp_dir, "data.yaml"), "w") as f:
                            yaml.dump(data_config, f)

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Create log file in current directory
                        log_file_path = f"training_log_{timestamp}.txt"
                        
                        # Determine which model to use based on training mode
                        safe_model_copy = None
                        if "best.pt" in training_mode:
                            # Use best.pt
                            if os.path.exists('best.pt'):
                                backup_path = f"temp_model_{timestamp}.pt"
                                shutil.copy2('best.pt', backup_path)
                                safe_model_copy = backup_path
                                st.info("🎯 Using best.pt for transfer learning")
                            else:
                                st.error("best.pt not found!")
                                st.stop()
                        elif "Transfer learning" in training_mode:
                            # Use currently selected model
                            if st.session_state.get('selected_model_path') and \
                                    os.path.exists(st.session_state.selected_model_path):
                                backup_path = f"temp_model_{timestamp}.pt"
                                shutil.copy2(st.session_state.selected_model_path, backup_path)
                                safe_model_copy = backup_path
                                st.info(f"🎯 Using {selected_name} for transfer learning")
                            else:
                                st.error("Selected model not found!")
                                st.stop()
                        elif "scratch" in training_mode.lower():
                            # Train from scratch with YOLO model
                            safe_model_copy = yolo_model
                            st.info(f"🎯 Training from scratch using {yolo_model}")

                        project_output = os.path.join(tmp_dir, "output")
                        os.makedirs(project_output, exist_ok=True)
                        run_name = f"st_run_{timestamp}"

                        cmd = [
                            sys.executable, "training_handler.py",
                            "--train_dir", tmp_dir,
                            "--epochs", str(epochs),
                            "--batch", str(batch),
                            "--imgsz", str(imgsz),
                            "--lr0", str(lr0),
                            "--lrf", str(lrf),
                            "--optimizer", str(optimizer),
                            "--weight_decay", str(weight_decay),
                            "--mosaic", str(use_mosaic),
                            "--degrees", str(degrees),
                            "--translate", str(translate),
                            "--scale", str(scale),
                            "--fliplr", str(fliplr),
                            "--project", project_output,
                            "--name", run_name,
                            "--patience", "50"
                        ]
                        if safe_model_copy:
                            cmd += ["--existing_model", safe_model_copy]
                        if use_clahe:
                            cmd += ["--use_clahe"]

                        candidate_best = os.path.join(project_output, run_name, "weights", "best.pt")
                        candidate_last = os.path.join(project_output, run_name, "weights", "last.pt")

                        # Create log file
                        with open(log_file_path, 'w') as f:
                            f.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"Training Mode: {training_mode}\n")
                            if "scratch" in training_mode.lower():
                                f.write(f"YOLO Model: {yolo_model}\n")
                            f.write(f"Command: {' '.join(cmd)}\n")
                            f.write("="*80 + "\n\n")

                        # Start process with output redirected to file
                        log_file_handle = open(log_file_path, 'a', buffering=1)
                        process = subprocess.Popen(
                            cmd, 
                            stdout=log_file_handle, 
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1
                        )

                        # Store in session state
                        st.session_state.training_process = process
                        st.session_state.training_active = True
                        st.session_state.training_logs = ''
                        st.session_state.training_result = None
                        st.session_state.training_safe_model_copy = safe_model_copy
                        st.session_state.training_candidate_best = candidate_best
                        st.session_state.training_candidate_last = candidate_last
                        st.session_state.training_log_file = log_file_path

                        # Background thread monitors process
                        def _monitor_training(proc, log_file, cand_best, cand_last, smc, log_handle):
                            try:
                                proc.wait()
                            except Exception:
                                pass
                            finally:
                                try:
                                    log_handle.close()
                                except:
                                    pass
                                
                                rc = proc.returncode
                                # Determine result FIRST before setting training_active
                                result_dict = None
                                if rc == 0 and os.path.exists(cand_best):
                                    next_v = get_next_model_version()
                                    new_name = f"training_{next_v}.pt"
                                    new_path = os.path.join(MODELS_DIR, new_name)
                                    shutil.copy2(cand_best, new_path)
                                    result_dict = {
                                        'status': 'success',
                                        'model_name': new_name,
                                        'model_path': new_path
                                    }
                                elif rc == 0 and os.path.exists(cand_last):
                                    next_v = get_next_model_version()
                                    new_name = f"training_{next_v}_last.pt"
                                    new_path = os.path.join(MODELS_DIR, new_name)
                                    shutil.copy2(cand_last, new_path)
                                    result_dict = {
                                        'status': 'partial',
                                        'model_name': new_name,
                                        'model_path': new_path
                                    }
                                else:
                                    result_dict = {
                                        'status': 'failed', 'rc': rc
                                    }
                                
                                # Set result FIRST
                                st.session_state.training_result = result_dict
                                
                                # Clean up temp backup (only if it's not a YOLO model name)
                                if smc and os.path.exists(smc) and not smc.startswith('yolov8'):
                                    try: os.remove(smc)
                                    except: pass
                                
                                # THEN mark training as inactive (this triggers UI update)
                                st.session_state.training_active = False

                        t = threading.Thread(
                            target=_monitor_training,
                            args=(process, log_file_path, candidate_best, candidate_last, safe_model_copy, log_file_handle),
                            daemon=True
                        )
                        t.start()
                        st.rerun()

            # ---- CLEAR TRAINING FILES ----
            st.markdown("---")
            st.markdown("#### 🧹 Clear Training Files")
            st.caption("Removes all temporary training directories, leftover model backups, and YOLO run folders from disk.")
            if st.button("🗑️ Clear All Training Files", type="secondary"):
                cleared = clear_training_files()
                if cleared:
                    st.success(f"✅ Cleared {len(cleared)} item(s):\n" + "\n".join(f"• `{c}`" for c in cleared))
                else:
                    st.info("Nothing to clear — no temporary training files found.")

    # ── Tab 3: Manage Models ─────────────────────────────────────────────────
    with tab3:
        st.markdown("### 🗂️ Model Management")
        available_models = get_available_models()
        if not available_models:
            st.info("No models found. Upload or train a model first.")
        else:
            st.write(f"**Total models:** {len(available_models)}")
            st.markdown("---")
            for idx, (path, name, mtime) in enumerate(available_models):
                with st.expander(f"📦 {name}", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**Path:** `{path}`")
                        st.write(f"**Size:** {os.path.getsize(path) / (1024 * 1024):.2f} MB")
                        st.write(f"**Modified:** {datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')}")
                        is_selected = (st.session_state.get('selected_model_path') == path)
                        if is_selected:
                            st.success("✅ Currently selected")
                    with col2:
                        if not is_selected:
                            if st.button(f"Use this model", key=f"use_{idx}"):
                                st.session_state.selected_model_idx = idx
                                st.session_state.selected_model_path = path
                                st.success(f"Now using {name}")
                                time.sleep(0.5)
                                st.rerun()
                        if st.button(f"🗑️ Delete", key=f"del_{idx}", type="secondary"):
                            if is_selected:
                                st.error("Cannot delete currently selected model")
                            else:
                                try:
                                    os.remove(path)
                                    # Reset selection to first available model after deletion
                                    st.session_state.selected_model_idx = 0
                                    remaining_models = get_available_models()
                                    if remaining_models:
                                        st.session_state.selected_model_path = remaining_models[0][0]
                                    else:
                                        st.session_state.selected_model_path = None
                                    st.success(f"Deleted {name}")
                                    time.sleep(0.5)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting: {e}")
                    st.markdown("---")

# ══════════════════════════════════════════════════════
# DATABASE PAGE
# ══════════════════════════════════════════════════════
elif page == "💾 Database":
    st.markdown('<p class="sub-header">Database</p>', unsafe_allow_html=True)
    try:
        conn = create_connection()
        if conn:
            st.success("✅ Connected")
            conn.close()
            tab1, tab2 = st.tabs(["📋 Defect Data", "🔬 Defect Info"])

            # ── TAB 1: defect_data ─────────────────────────────────────────
            with tab1:
                data = fetch_all_entries()
                if not data:
                    st.info("No records in defect_data.")
                else:
                    df_data = pd.DataFrame(data)
                    st.write(f"**Total records:** {len(df_data)}")
                    search_term = st.text_input("🔍 Search (any column)", key="search_data",
                                               placeholder="Type to filter…")
                    if search_term:
                        mask = df_data.apply(
                            lambda col: col.astype(str).str.contains(search_term, case=False, na=False)
                        ).any(axis=1)
                        df_data = df_data[mask]
                        st.caption(f"Showing {len(df_data)} matching record(s)")
                    st.dataframe(df_data, use_container_width=True)
                    st.markdown("---")

                    st.markdown("#### ✏️ Edit Record")
                    if 'id' in df_data.columns:
                        edit_id = st.selectbox("Select record ID to edit", df_data['id'].tolist(),
                                               key="edit_data_id")
                        row = df_data[df_data['id'] == edit_id].iloc[0]
                        editable_cols = [c for c in df_data.columns if c != 'id']
                        with st.form("edit_data_form"):
                            edited_vals = {}
                            col_pairs = [editable_cols[i:i+2] for i in range(0, len(editable_cols), 2)]
                            for pair in col_pairs:
                                fcols = st.columns(len(pair))
                                for fi, col_name in enumerate(pair):
                                    with fcols[fi]:
                                        edited_vals[col_name] = st.text_input(
                                            col_name,
                                            value=str(row[col_name]) if pd.notna(row[col_name]) else "",
                                            key=f"edit_data_{col_name}"
                                        )
                            if st.form_submit_button("💾 Save Changes", type="primary"):
                                ok = update_defect_data(edit_id, edited_vals)
                                if ok:
                                    st.success(f"✅ Record ID {edit_id} updated!")
                                    time.sleep(0.5); st.rerun()
                                else:
                                    st.error("❌ Update failed.")
                    else:
                        st.info("No 'id' column found — edit unavailable.")

                    st.markdown("---")
                    st.markdown("#### 🗑️ Delete Record")
                    if 'id' in df_data.columns:
                        del_col1, del_col2 = st.columns([3, 1])
                        with del_col1:
                            delete_id = st.selectbox("Select record ID to delete",
                                                     df_data['id'].tolist(), key="del_data_id")
                        with del_col2:
                            st.write("")
                            confirm_del = st.checkbox("Confirm", key="confirm_del_data")
                        if st.button("🗑️ Delete Record", type="secondary", key="btn_del_data"):
                            if confirm_del:
                                ok = delete_defect_data(delete_id)
                                if ok:
                                    st.success(f"✅ Record ID {delete_id} deleted!")
                                    time.sleep(0.5); st.rerun()
                                else:
                                    st.error("❌ Delete failed.")
                            else:
                                st.warning("⚠️ Tick 'Confirm' to proceed.")

                    st.markdown("---")
                    st.markdown("#### 🧹 Bulk Delete")
                    if 'id' in df_data.columns:
                        bulk_ids = st.multiselect("Select multiple IDs to delete",
                                                  df_data['id'].tolist(), key="bulk_del_data")
                        confirm_bulk = st.checkbox("Confirm bulk delete", key="confirm_bulk_data")
                        if st.button("🗑️ Delete Selected", type="secondary", key="btn_bulk_del_data"):
                            if not bulk_ids:
                                st.warning("No records selected.")
                            elif not confirm_bulk:
                                st.warning("⚠️ Tick 'Confirm bulk delete' to proceed.")
                            else:
                                errors = [rid for rid in bulk_ids if not delete_defect_data(rid)]
                                if errors:
                                    st.error(f"❌ Failed to delete IDs: {errors}")
                                else:
                                    st.success(f"✅ Deleted {len(bulk_ids)} record(s)!")
                                time.sleep(0.5); st.rerun()

            # ── TAB 2: defect_info ─────────────────────────────────────────
            with tab2:
                info = fetch_all_defect_info()
                if not info:
                    st.info("No records in defect_info.")
                else:
                    df_info = pd.DataFrame(info)
                    st.write(f"**Total records:** {len(df_info)}")
                    search_info = st.text_input("🔍 Search (any column)", key="search_info",
                                               placeholder="Type to filter…")
                    if search_info:
                        mask = df_info.apply(
                            lambda col: col.astype(str).str.contains(search_info, case=False, na=False)
                        ).any(axis=1)
                        df_info = df_info[mask]
                        st.caption(f"Showing {len(df_info)} matching record(s)")
                    st.dataframe(df_info, use_container_width=True)
                    st.markdown("---")

                    st.markdown("#### ✏️ Edit Record")
                    if 'id' in df_info.columns:
                        edit_info_id = st.selectbox("Select record ID to edit",
                                                    df_info['id'].tolist(), key="edit_info_id")
                        row_i = df_info[df_info['id'] == edit_info_id].iloc[0]
                        editable_cols_i = [c for c in df_info.columns if c != 'id']
                        with st.form("edit_info_form"):
                            edited_vals_i = {}
                            col_pairs_i = [editable_cols_i[i:i+2] for i in range(0, len(editable_cols_i), 2)]
                            for pair in col_pairs_i:
                                fcols = st.columns(len(pair))
                                for fi, col_name in enumerate(pair):
                                    with fcols[fi]:
                                        edited_vals_i[col_name] = st.text_area(
                                            col_name,
                                            value=str(row_i[col_name]) if pd.notna(row_i[col_name]) else "",
                                            key=f"edit_info_{col_name}", height=80
                                        )
                            if st.form_submit_button("💾 Save Changes", type="primary"):
                                ok = update_defect_info_record(edit_info_id, edited_vals_i)
                                if ok:
                                    st.success(f"✅ Record ID {edit_info_id} updated!")
                                    time.sleep(0.5); st.rerun()
                                else:
                                    st.error("❌ Update failed.")
                    else:
                        st.info("No 'id' column found — edit unavailable.")

                    st.markdown("---")
                    st.markdown("#### 🗑️ Delete Record")
                    if 'id' in df_info.columns:
                        del_col1i, del_col2i = st.columns([3, 1])
                        with del_col1i:
                            delete_info_id = st.selectbox("Select record ID to delete",
                                                          df_info['id'].tolist(), key="del_info_id")
                        with del_col2i:
                            st.write("")
                            confirm_del_i = st.checkbox("Confirm", key="confirm_del_info")
                        if st.button("🗑️ Delete Record", type="secondary", key="btn_del_info"):
                            if confirm_del_i:
                                ok = delete_defect_info_record(delete_info_id)
                                if ok:
                                    st.success(f"✅ Record ID {delete_info_id} deleted!")
                                    time.sleep(0.5); st.rerun()
                                else:
                                    st.error("❌ Delete failed.")
                            else:
                                st.warning("⚠️ Tick 'Confirm' to proceed.")

                    st.markdown("---")
                    st.markdown("#### 🧹 Bulk Delete")
                    if 'id' in df_info.columns:
                        bulk_ids_i = st.multiselect("Select multiple IDs to delete",
                                                    df_info['id'].tolist(), key="bulk_del_info")
                        confirm_bulk_i = st.checkbox("Confirm bulk delete", key="confirm_bulk_info")
                        if st.button("🗑️ Delete Selected", type="secondary", key="btn_bulk_del_info"):
                            if not bulk_ids_i:
                                st.warning("No records selected.")
                            elif not confirm_bulk_i:
                                st.warning("⚠️ Tick 'Confirm bulk delete' to proceed.")
                            else:
                                errors_i = [rid for rid in bulk_ids_i if not delete_defect_info_record(rid)]
                                if errors_i:
                                    st.error(f"❌ Failed to delete IDs: {errors_i}")
                                else:
                                    st.success(f"✅ Deleted {len(bulk_ids_i)} record(s)!")
                                time.sleep(0.5); st.rerun()
        else:
            st.error("Cannot connect to database.")
    except Exception as e:
        st.error(f"Error: {e}")
        traceback.print_exc()

# ══════════════════════════════════════════════════════
# HELP PAGE
# ══════════════════════════════════════════════════════
else:
    st.markdown('<p class="sub-header">Help</p>', unsafe_allow_html=True)
    st.markdown("""
    ## Quick Start
    **Detection:** Select model → Upload image → Analyze → Save to DB
    **Processing:** Upload → Select filters → Apply
    **Training:** Choose mode → Upload data → Configure → Start Training
    **Database:** View, search, edit, or delete records

    ## Training Modes 🎯
    
    ### 1. Use best.pt (Legacy Model)
    - Continue training from your old best.pt file
    - Only appears if best.pt exists in your project directory
    - Fast and efficient for improving existing models
    
    ### 2. Transfer Learning
    - Fine-tune from your currently selected model
    - Shown with the model name (e.g., "Transfer learning from training_1.pt")
    - Best for iterative improvements
    
    ### 3. Train from Scratch
    - Start fresh with a new YOLO model architecture
    - Choose from 5 variants:
      - **Nano** (yolov8n.pt) - ~6MB, fastest, good for quick testing
      - **Small** (yolov8s.pt) - ~22MB, balanced performance (default)
      - **Medium** (yolov8m.pt) - ~52MB, better accuracy
      - **Large** (yolov8l.pt) - ~88MB, high accuracy
      - **XLarge** (yolov8x.pt) - ~136MB, maximum accuracy
    - Models auto-download from Ultralytics on first use
    - Useful when you want to try different architectures

    ## Training Process 🎓
    - Upload image/label pairs — training is blocked if matched pairs < minimum
    - Training runs in background — UI stays responsive
    - **Real-time logs** saved to file and displayed live
    - **Auto-completion**: UI automatically refreshes when training finishes
    - Training status shown in sidebar on all pages
    - Use **🛑 Stop Training** anytime to cancel
    - Use **🗑️ Clear All Training Files** to free disk space

    ## Training Output
    - Progress bar shows current epoch and live loss values
    - Log panel shows last 60 clean lines (all terminal noise stripped)
    - Logs captured from file - no threading issues
    - Models auto-saved with version numbers (training_1.pt, training_2.pt, etc.)

    ## Database Management 💾
    - **Search:** Filter records by any column in real time
    - **Edit:** Select record ID and modify fields inline
    - **Delete:** Select record ID + confirm, then delete
    - **Bulk Delete:** Multiselect IDs + confirm, then delete all

    ## Label Format (YOLO)
    ```
    class_id cx cy w h
    ```
    Classes: 0=LOP, 1=linear pore, 2=porosity

    ## Tips
    - Models stored in `models/` directory
    - Training logs saved to `training_log_TIMESTAMP.txt` files
    - YOLO models cached in `~/.cache/ultralytics/` after download
    - You can train multiple versions and switch between them freely
    - Smaller models train faster but may be less accurate
    - Larger models are more accurate but slower to train
    """)

st.markdown("---")
st.markdown("<div style='text-align:center;color:gray;'>X-ray Detection v3.5 - Improved Auto-completion & Debugging</div>", unsafe_allow_html=True)