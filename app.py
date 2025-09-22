import streamlit as st
import cv2
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io
import zipfile
import tempfile
import os
from typing import Dict, Tuple, List
import re

# Configure page
st.set_page_config(
    page_title="Insurance Document Extractor",
    page_icon="📄",
    layout="wide"
)

# Updated coordinates based on your document layout (at 300 DPI)
DEFAULT_COORDINATES = {
    'property_coverage': (645, 1725, 1050, 1770),    # Red box - $261,783.00
    'loss_of_rent': (645, 1810, 1050, 1855),        # Green box - $22,800.00  
    'total': (530, 2320, 830, 2365),                # Blue box - $945.59
    'location_description': (225, 1200, 1050, 1290) # Yellow box - address area
}

class PDFExtractor:
    def __init__(self, field_coordinates, extraction_dpi=300):
        self.results = []
        self.field_coordinates = field_coordinates
        self.extraction_dpi = extraction_dpi
    
    def preprocess_image_pil(self, image_pil: Image.Image) -> Image.Image:
        """Preprocess image using PIL"""
        try:
            # Convert to grayscale
            gray = image_pil.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(gray)
            enhanced = enhancer.enhance(1.5)
            
            # Enhance sharpness
            sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
            sharpened = sharpness_enhancer.enhance(2.0)
            
            return sharpened
        except Exception as e:
            return image_pil.convert('L')
    
    def preprocess_image_cv2(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image using OpenCV"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Apply adaptive thresholding for better text detection
            adaptive_thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            return adaptive_thresh
        except Exception:
            return image
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy"""
        # Try OpenCV first
        try:
            return self.preprocess_image_cv2(image)
        except Exception:
            pass
        
        # Fallback to PIL
        try:
            image_pil = Image.fromarray(image)
            processed_pil = self.preprocess_image_pil(image_pil)
            return np.array(processed_pil)
        except Exception:
            return image
    
    def is_valid_extraction(self, text: str, field_name: str) -> bool:
        """Check if extraction result is valid"""
        if not text or text.startswith("No text") or text.startswith("OCR error") or text.startswith("Invalid") or text.startswith("Coordinates"):
            return False
        
        # For monetary fields, check if we have digits
        if any(keyword in field_name.lower() for keyword in ['coverage', 'rent', 'total']):
            # Must contain at least one digit
            return bool(re.search(r'\d', text))
        
        # For location/description, check minimum length
        if 'location' in field_name.lower() or 'description' in field_name.lower():
            return len(text.strip()) > 3
        
        return True
    
    def extract_field_value(self, image: np.ndarray, coordinates: Tuple[int, int, int, int], field_name: str) -> str:
        """Extract text from specific coordinates"""
        x1, y1, x2, y2 = coordinates
        
        # Validate coordinates
        if x1 >= x2 or y1 >= y2:
            return f"Invalid coordinates for {field_name}"
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        
        if x1 >= x2 or y1 >= y2:
            return f"Coordinates out of bounds for {field_name}"
        
        # Extract region of interest
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return f"Empty ROI for {field_name}"
        
        # Preprocess ROI
        processed_roi = self.preprocess_image(roi)
        
        # Configure OCR based on field type
        if any(keyword in field_name.lower() for keyword in ['coverage', 'rent', 'total']):
            # For monetary amounts
            config = '--psm 8 -c tessedit_char_whitelist=0123456789.,$'
        elif 'location' in field_name.lower() or 'description' in field_name.lower():
            # For addresses
            config = '--psm 6'
        else:
            # General text
            config = '--psm 7'
        
        try:
            # Perform OCR
            text = pytesseract.image_to_string(processed_roi, config=config).strip()
            
            # Clean up the text based on field type
            if any(keyword in field_name.lower() for keyword in ['coverage', 'rent', 'total']):
                # For monetary amounts
                text = re.sub(r'[^\d.,]', '', text)
                text = text.strip('.,')
                if text.count('.') > 1:
                    parts = text.split('.')
                    text = '.'.join(parts[:2])
            elif 'location' in field_name.lower() or 'description' in field_name.lower():
                # For location/description
                text = ' '.join(text.split())
                text = re.sub(r'[|_\\]', '', text)
            
            return text if text else f"No text detected in {field_name}"
            
        except Exception as e:
            return f"OCR error in {field_name}: {str(e)}"
    
    def apply_y_offset(self, coordinates: Dict[str, Tuple[int, int, int, int]], y_offset: int) -> Dict[str, Tuple[int, int, int, int]]:
        """Apply Y offset to all coordinates"""
        offset_coordinates = {}
        for field_name, (x1, y1, x2, y2) in coordinates.items():
            offset_coordinates[field_name] = (x1, y1 + y_offset, x2, y2 + y_offset)
        return offset_coordinates
    
    def extract_all_fields_with_coordinates(self, image: np.ndarray, coordinates: Dict[str, Tuple[int, int, int, int]]) -> Dict[str, str]:
        """Extract all fields using given coordinates"""
        results = {}
        for field_name, coords in coordinates.items():
            value = self.extract_field_value(image, coords, field_name)
            results[field_name] = value
        return results
    
    def count_valid_extractions(self, extraction_results: Dict[str, str]) -> int:
        """Count how many fields were successfully extracted"""
        valid_count = 0
        for field_name, value in extraction_results.items():
            if self.is_valid_extraction(value, field_name):
                valid_count += 1
        return valid_count
    
    def extract_with_fallback(self, image: np.ndarray, coordinates: Dict[str, Tuple[int, int, int, int]], filename: str) -> Dict[str, str]:
        """Extract fields with automatic fallback using Y-offset, choosing the method with most successful extractions"""
        
        # First attempt with original coordinates
        original_results = self.extract_all_fields_with_coordinates(image, coordinates)
        original_valid_count = self.count_valid_extractions(original_results)
        
        # Check if property coverage extraction failed (trigger for trying offset)
        property_coverage = original_results.get('property_coverage', '')
        
        if not self.is_valid_extraction(property_coverage, 'property_coverage'):
            st.info(f"🔄 Property coverage not found in {filename}, trying with Y+45 offset...")
            
            # Apply Y offset and try again
            offset_coordinates = self.apply_y_offset(coordinates, 45)
            offset_results = self.extract_all_fields_with_coordinates(image, offset_coordinates)
            offset_valid_count = self.count_valid_extractions(offset_results)
            
            # Choose the method that yielded more successful extractions
            if offset_valid_count > original_valid_count:
                st.success(f"✅ Y+45 offset method yielded {offset_valid_count} vs {original_valid_count} successful extractions for {filename}")
                final_results = offset_results.copy()
                extraction_method = "Y+45_offset"
            else:
                st.info(f"📊 Original method still better: {original_valid_count} vs {offset_valid_count} successful extractions for {filename}")
                final_results = original_results.copy()
                extraction_method = "original"
        else:
            # Original method worked for property coverage
            final_results = original_results.copy()
            extraction_method = "original"
        
        # Add filename and overall extraction method
        final_results['filename'] = filename
        final_results['extraction_method'] = extraction_method
        
        # Add individual field success indicators
        for field_name in coordinates.keys():
            field_success = self.is_valid_extraction(final_results[field_name], field_name)
            final_results[f"{field_name}_success"] = "✅" if field_success else "❌"
        
        return final_results
    
    def process_pdf_from_bytes(self, pdf_bytes: bytes, filename: str) -> Dict[str, str]:
        """Process PDF from bytes"""
        try:
            # Convert PDF to images at extraction DPI
            images = convert_from_bytes(pdf_bytes, dpi=self.extraction_dpi)
            
            if not images:
                return {"error": "No pages found in PDF", "filename": filename}
            
            # Convert PIL image to numpy array
            image_np = np.array(images[0])
            
            # Extract all fields with fallback
            extracted_data = self.extract_with_fallback(image_np, self.field_coordinates, filename)
            
            return extracted_data
            
        except Exception as e:
            return {"error": f"Error processing PDF: {str(e)}", "filename": filename}
    
    def process_pdf(self, pdf_file) -> Dict[str, str]:
        """Process a single PDF file"""
        try:
            if hasattr(pdf_file, 'read'):
                # Streamlit uploaded file
                pdf_bytes = pdf_file.read()
                filename = pdf_file.name
            else:
                # File path
                with open(pdf_file, 'rb') as f:
                    pdf_bytes = f.read()
                filename = os.path.basename(pdf_file)
            
            return self.process_pdf_from_bytes(pdf_bytes, filename)
            
        except Exception as e:
            return {"error": f"Error processing PDF: {str(e)}", "filename": getattr(pdf_file, 'name', 'unknown')}
    
    def extract_pdfs_from_zip(self, zip_file) -> List[Tuple[str, bytes]]:
        """Extract PDF files from ZIP archive"""
        pdf_files = []
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                for file_info in zip_ref.filelist:
                    if file_info.filename.lower().endswith('.pdf') and not file_info.filename.startswith('__MACOSX'):
                        pdf_bytes = zip_ref.read(file_info.filename)
                        # Clean filename
                        filename = os.path.basename(file_info.filename)
                        pdf_files.append((filename, pdf_bytes))
                        
        except Exception as e:
            st.error(f"Error extracting ZIP file: {str(e)}")
            
        return pdf_files
    
    def process_zip_file(self, zip_file) -> pd.DataFrame:
        """Process ZIP file containing PDFs"""
        # Extract PDFs from ZIP
        pdf_files = self.extract_pdfs_from_zip(zip_file)
        
        if not pdf_files:
            st.error("No PDF files found in ZIP archive")
            return pd.DataFrame()
        
        st.success(f"Found {len(pdf_files)} PDF files in ZIP archive")
        
        results = []
        progress_bar = st.progress(0)
        
        for i, (filename, pdf_bytes) in enumerate(pdf_files):
            st.write(f"Processing: {filename}")
            
            # Process PDF
            extracted_data = self.process_pdf_from_bytes(pdf_bytes, filename)
            results.append(extracted_data)
            
            # Update progress
            progress_bar.progress((i + 1) / len(pdf_files))
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Reorder columns - main fields first, then success indicators, then method info
        main_fields = ['filename', 'property_coverage', 'loss_of_rent', 'total', 'location_description']
        success_fields = [col for col in df.columns if col.endswith('_success')]
        method_fields = ['extraction_method']
        other_fields = [col for col in df.columns if col not in main_fields and col not in success_fields and col not in method_fields]
        
        column_order = main_fields + success_fields + method_fields + other_fields
        existing_cols = [col for col in column_order if col in df.columns]
        df = df[existing_cols]
        
        return df
    
    def process_multiple_pdfs(self, pdf_files: List) -> pd.DataFrame:
        """Process multiple PDF files"""
        results = []
        progress_bar = st.progress(0)
        
        for i, pdf_file in enumerate(pdf_files):
            st.write(f"Processing: {pdf_file.name}")
            
            # Extract data from PDF
            extracted_data = self.process_pdf(pdf_file)
            results.append(extracted_data)
            
            # Update progress
            progress_bar.progress((i + 1) / len(pdf_files))
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Reorder columns - main fields first, then success indicators, then method info
        main_fields = ['filename', 'property_coverage', 'loss_of_rent', 'total', 'location_description']
        success_fields = [col for col in df.columns if col.endswith('_success')]
        method_fields = ['extraction_method']
        other_fields = [col for col in df.columns if col not in main_fields and col not in success_fields and col not in method_fields]
        
        column_order = main_fields + success_fields + method_fields + other_fields
        existing_cols = [col for col in column_order if col in df.columns]
        df = df[existing_cols]
        
        return df

def get_field_coordinates_from_sidebar():
    """Get field coordinates from sidebar inputs"""
    coordinates = {}
    
    st.sidebar.title("⚙️ Field Coordinates")
    st.sidebar.markdown("**Coordinates are for 300 DPI extraction**")
    st.sidebar.markdown("Adjust extraction coordinates for each field:")
    
    # Show coordinate tips
    with st.sidebar.expander("📏 Coordinate Tips", expanded=False):
        st.markdown("""
        **Updated coordinates for your document:**
        - Property Coverage: (645, 1725, 1050, 1770)
        - Loss of Rent: (645, 1810, 1050, 1855) ✅ **Updated**
        - Total: (530, 2320, 830, 2365)  
        - Location: (225, 1200, 1050, 1290)
        
        **Smart fallback:** Uses method with most successful extractions
        """)
    
    # Property Coverage
    with st.sidebar.expander("📊 Property Coverage", expanded=True):
        pc_x1 = st.number_input("X1 (left)", value=DEFAULT_COORDINATES['property_coverage'][0], key="pc_x1")
        pc_y1 = st.number_input("Y1 (top)", value=DEFAULT_COORDINATES['property_coverage'][1], key="pc_y1")
        pc_x2 = st.number_input("X2 (right)", value=DEFAULT_COORDINATES['property_coverage'][2], key="pc_x2")
        pc_y2 = st.number_input("Y2 (bottom)", value=DEFAULT_COORDINATES['property_coverage'][3], key="pc_y2")
        coordinates['property_coverage'] = (pc_x1, pc_y1, pc_x2, pc_y2)
    
    # Loss of Rent
    with st.sidebar.expander("🏠 Loss of Rent", expanded=False):
        lr_x1 = st.number_input("X1 (left)", value=DEFAULT_COORDINATES['loss_of_rent'][0], key="lr_x1")
        lr_y1 = st.number_input("Y1 (top)", value=DEFAULT_COORDINATES['loss_of_rent'][1], key="lr_y1")
        lr_x2 = st.number_input("X2 (right)", value=DEFAULT_COORDINATES['loss_of_rent'][2], key="lr_x2")
        lr_y2 = st.number_input("Y2 (bottom)", value=DEFAULT_COORDINATES['loss_of_rent'][3], key="lr_y2")
        coordinates['loss_of_rent'] = (lr_x1, lr_y1, lr_x2, lr_y2)
    
    # Total
    with st.sidebar.expander("💰 Total", expanded=False):
        total_x1 = st.number_input("X1 (left)", value=DEFAULT_COORDINATES['total'][0], key="total_x1")
        total_y1 = st.number_input("Y1 (top)", value=DEFAULT_COORDINATES['total'][1], key="total_y1")
        total_x2 = st.number_input("X2 (right)", value=DEFAULT_COORDINATES['total'][2], key="total_x2")
        total_y2 = st.number_input("Y2 (bottom)", value=DEFAULT_COORDINATES['total'][3], key="total_y2")
        coordinates['total'] = (total_x1, total_y1, total_x2, total_y2)
    
    # Location/Description
    with st.sidebar.expander("📍 Location/Description", expanded=False):
        loc_x1 = st.number_input("X1 (left)", value=DEFAULT_COORDINATES['location_description'][0], key="loc_x1")
        loc_y1 = st.number_input("Y1 (top)", value=DEFAULT_COORDINATES['location_description'][1], key="loc_y1")
        loc_x2 = st.number_input("X2 (right)", value=DEFAULT_COORDINATES['location_description'][2], key="loc_x2")
        loc_y2 = st.number_input("Y2 (bottom)", value=DEFAULT_COORDINATES['location_description'][3], key="loc_y2")
        coordinates['location_description'] = (loc_x1, loc_y1, loc_x2, loc_y2)
    
    # Reset to defaults button
    if st.sidebar.button("🔄 Reset to Defaults"):
        st.rerun()
    
    return coordinates

def draw_rectangles_safe(image_np, field_coordinates, preview_dpi=200):
    """Draw rectangles with DPI scaling for preview"""
    # Scale coordinates from 300 DPI (extraction) to preview DPI
    scaled_coordinates = {}
    for field_name, coords in field_coordinates.items():
        x1, y1, x2, y2 = coords
        scale_factor = preview_dpi / 300
        scaled_coordinates[field_name] = (
            int(x1 * scale_factor),
            int(y1 * scale_factor),
            int(x2 * scale_factor), 
            int(y2 * scale_factor)
        )
    
    try:
        # Use OpenCV
        colors = {
            'property_coverage': (255, 0, 0),      # Red
            'loss_of_rent': (0, 255, 0),          # Green
            'total': (0, 0, 255),                 # Blue
            'location_description': (255, 255, 0)  # Yellow
        }
        
        for field_name, (x1, y1, x2, y2) in scaled_coordinates.items():
            color = colors.get(field_name, (255, 0, 0))
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 3)
            
            # Add field label
            label = field_name.replace('_', ' ').title()
            cv2.putText(image_np, label, (x1, y1-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return image_np
    except Exception:
        # Fallback: use PIL
        try:
            from PIL import ImageDraw
            
            image_pil = Image.fromarray(image_np)
            draw = ImageDraw.Draw(image_pil)
            
            colors = {
                'property_coverage': (255, 0, 0),
                'loss_of_rent': (0, 255, 0),
                'total': (0, 0, 255),
                'location_description': (255, 255, 0)
            }
            
            for field_name, (x1, y1, x2, y2) in scaled_coordinates.items():
                color = colors.get(field_name, (255, 0, 0))
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                label = field_name.replace('_', ' ').title()
                draw.text((x1, y1-20), label, fill=color)
            
            return np.array(image_pil)
        except Exception:
            return image_np

def main():
    st.title("📄 Insurance Document Data Extractor")
    st.markdown("Upload scanned PDF insurance documents to extract key information")
    
    # Initialize session state
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = None
    if 'uploaded_zip' not in st.session_state:
        st.session_state.uploaded_zip = None
    if 'upload_method' not in st.session_state:
        st.session_state.upload_method = "📄 Individual PDF Files"
    
    # Get field coordinates from sidebar
    field_coordinates = get_field_coordinates_from_sidebar()
    
    # Additional sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Smart Extraction")
    st.sidebar.markdown("""
    - **Best method selection** based on total success count
    - **Updated Loss of Rent coordinates** ✅
    - **Success indicators** for each field
    - Red: Property Coverage
    - Green: Loss of Rent  
    - Blue: Total
    - Yellow: Location
    """)
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Extract", "🔍 Preview", "📊 Results"])
    
    with tab1:
        st.header("Upload Files")
        
        # File upload options
        upload_option = st.radio(
            "Choose upload method:",
            ["📄 Individual PDF Files", "📦 ZIP File containing PDFs"],
            horizontal=True
        )
        
        # Store upload method in session state
        st.session_state.upload_method = upload_option
        
        if upload_option == "📄 Individual PDF Files":
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload one or more PDF files with identical layouts"
            )
            
            # Store in session state
            if uploaded_files:
                st.session_state.uploaded_files = uploaded_files
                st.session_state.uploaded_zip = None  # Clear ZIP if PDFs uploaded
            
            if st.session_state.uploaded_files:
                st.success(f"Uploaded {len(st.session_state.uploaded_files)} PDF file(s)")
                
                # Show current coordinates
                with st.expander("📐 Current Coordinates (300 DPI)", expanded=False):
                    coord_df = pd.DataFrame([
                        {"Field": "Property Coverage", "X1": field_coordinates['property_coverage'][0], 
                         "Y1": field_coordinates['property_coverage'][1], "X2": field_coordinates['property_coverage'][2], 
                         "Y2": field_coordinates['property_coverage'][3]},
                        {"Field": "Loss of Rent", "X1": field_coordinates['loss_of_rent'][0], 
                         "Y1": field_coordinates['loss_of_rent'][1], "X2": field_coordinates['loss_of_rent'][2], 
                         "Y2": field_coordinates['loss_of_rent'][3]},
                        {"Field": "Total", "X1": field_coordinates['total'][0], 
                         "Y1": field_coordinates['total'][1], "X2": field_coordinates['total'][2], 
                         "Y2": field_coordinates['total'][3]},
                        {"Field": "Location/Description", "X1": field_coordinates['location_description'][0], 
                         "Y1": field_coordinates['location_description'][1], "X2": field_coordinates['location_description'][2], 
                         "Y2": field_coordinates['location_description'][3]}
                    ])
                    st.dataframe(coord_df, hide_index=True)
                
                # Extract button
                if st.button("🚀 Extract Data from PDFs", type="primary"):
                    extractor = PDFExtractor(field_coordinates, extraction_dpi=300)
                    
                    with st.spinner("Processing PDF files..."):
                        results_df = extractor.process_multiple_pdfs(st.session_state.uploaded_files)
                    
                    # Store results in session state
                    st.session_state['results_df'] = results_df
                    st.success("✅ Extraction completed!")
                    
                    # Show summary of extraction methods used
                    if 'extraction_method' in results_df.columns:
                        method_counts = results_df['extraction_method'].value_counts()
                        st.subheader("📈 Extraction Method Summary")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if 'original' in method_counts:
                                st.metric("Files using Original Coordinates", method_counts.get('original', 0))
                        with col2:
                            if 'Y+45_offset' in method_counts:
                                st.metric("Files using Y+45 Offset", method_counts.get('Y+45_offset', 0))
                    
                    # Show preview
                    st.subheader("📋 Extraction Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download button
                    csv_data = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name="extracted_insurance_data.csv",
                        mime="text/csv"
                    )
        
        else:  # ZIP file option
            uploaded_zip = st.file_uploader(
                "Choose ZIP file containing PDFs",
                type=['zip'],
                help="Upload a ZIP file containing PDF documents with identical layouts"
            )
            
            # Store in session state
            if uploaded_zip:
                st.session_state.uploaded_zip = uploaded_zip
                st.session_state.uploaded_files = None  # Clear PDFs if ZIP uploaded
            
            if st.session_state.uploaded_zip:
                st.success(f"Uploaded ZIP file: {st.session_state.uploaded_zip.name}")
                
                # Show current coordinates
                with st.expander("📐 Current Coordinates (300 DPI)", expanded=False):
                    coord_df = pd.DataFrame([
                        {"Field": "Property Coverage", "X1": field_coordinates['property_coverage'][0], 
                         "Y1": field_coordinates['property_coverage'][1], "X2": field_coordinates['property_coverage'][2], 
                         "Y2": field_coordinates['property_coverage'][3]},
                        {"Field": "Loss of Rent", "X1": field_coordinates['loss_of_rent'][0], 
                         "Y1": field_coordinates['loss_of_rent'][1], "X2": field_coordinates['loss_of_rent'][2], 
                         "Y2": field_coordinates['loss_of_rent'][3]},
                        {"Field": "Total", "X1": field_coordinates['total'][0], 
                         "Y1": field_coordinates['total'][1], "X2": field_coordinates['total'][2], 
                         "Y2": field_coordinates['total'][3]},
                        {"Field": "Location/Description", "X1": field_coordinates['location_description'][0], 
                         "Y1": field_coordinates['location_description'][1], "X2": field_coordinates['location_description'][2], 
                         "Y2": field_coordinates['location_description'][3]}
                    ])
                    st.dataframe(coord_df, hide_index=True)
                
                # Extract button
                if st.button("🚀 Extract Data from ZIP", type="primary"):
                    extractor = PDFExtractor(field_coordinates, extraction_dpi=300)
                    
                    with st.spinner("Processing ZIP file..."):
                        results_df = extractor.process_zip_file(st.session_state.uploaded_zip)
                    
                    if not results_df.empty:
                        # Store results in session state
                        st.session_state['results_df'] = results_df
                        st.success("✅ Extraction completed!")
                        
                        # Show summary of extraction methods used
                        if 'extraction_method' in results_df.columns:
                            method_counts = results_df['extraction_method'].value_counts()
                            st.subheader("📈 Extraction Method Summary")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if 'original' in method_counts:
                                    st.metric("Files using Original Coordinates", method_counts.get('original', 0))
                            with col2:
                                if 'Y+45_offset' in method_counts:
                                    st.metric("Files using Y+45 Offset", method_counts.get('Y+45_offset', 0))
                        
                        # Show preview
                        st.subheader("📋 Extraction Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download button
                        csv_data = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name="extracted_insurance_data.csv",
                            mime="text/csv"
                        )
    
    with tab2:
        st.header("🔍 Preview Extraction Areas")
        st.info("Preview shows coordinates scaled to 200 DPI for faster loading")
        
        # Get files from session state
        files_to_preview = None
        
        if st.session_state.upload_method == "📄 Individual PDF Files" and st.session_state.uploaded_files:
            files_to_preview = st.session_state.uploaded_files
            
        elif st.session_state.upload_method == "📦 ZIP File containing PDFs" and st.session_state.uploaded_zip:
            # Extract one PDF from ZIP for preview
            extractor = PDFExtractor(field_coordinates)
            pdf_files = extractor.extract_pdfs_from_zip(st.session_state.uploaded_zip)
            if pdf_files:
                # Create a mock file object for preview
                class MockFile:
                    def __init__(self, name, content):
                        self.name = name
                        self.content = content
                    def read(self):
                        return self.content
                
                files_to_preview = [MockFile(name, content) for name, content in pdf_files[:3]]  # Show up to 3 files
        
        if files_to_preview:
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.subheader("Field Colors")
                st.markdown("""
                - 🔴 **Property Coverage**
                - 🟢 **Loss of Rent** ✅ **Updated**
                - 🔵 **Total**
                - 🟡 **Location/Description**
                
                **Current 300 DPI Coordinates:**
                """)
                
                st.text(f"Property: {field_coordinates['property_coverage']}")
                st.text(f"Loss Rent: {field_coordinates['loss_of_rent']}")  
                st.text(f"Total: {field_coordinates['total']}")
                st.text(f"Location: {field_coordinates['location_description']}")
                
                # File selector
                if len(files_to_preview) > 1:
                    selected_file = st.selectbox(
                        "Select file to preview:", 
                        files_to_preview, 
                        format_func=lambda x: x.name
                    )
                else:
                    selected_file = files_to_preview[0]
                    st.write(f"**Previewing:** {selected_file.name}")
            
            with col1:
                if st.button("👁️ Show Extraction Areas"):
                    try:
                        # Convert first page to image at 200 DPI for preview
                        pdf_bytes = selected_file.read()
                        images = convert_from_bytes(pdf_bytes, dpi=200)
                        
                        if images:
                            image_np = np.array(images[0])
                            
                            # Draw rectangles with DPI scaling
                            image_with_boxes = draw_rectangles_safe(image_np, field_coordinates, preview_dpi=200)
                            
                            # Display image
                            st.image(image_with_boxes, caption="Extraction Areas (200 DPI Preview)", use_column_width=True)
                            
                            # Show extracted values using actual extraction coordinates
                            st.subheader("🔍 Extracted Values Preview")
                            extractor = PDFExtractor(field_coordinates, extraction_dpi=300)
                            extracted = extractor.process_pdf_from_bytes(pdf_bytes, selected_file.name)
                            
                            if 'error' not in extracted:
                                # Separate main fields from success/method fields
                                main_data = {k: v for k, v in extracted.items() if not k.endswith('_success') and k != 'extraction_method'}
                                
                                preview_df = pd.DataFrame([main_data])
                                st.dataframe(preview_df, hide_index=True)
                                
                                # Show extraction info
                                if 'extraction_method' in extracted:
                                    method = extracted['extraction_method']
                                    if method == 'Y+45_offset':
                                        st.info(f"🔄 Used Y+45 offset method for this document")
                                    else:
                                        st.success(f"✅ Used original coordinates for this document")
                                
                                # Show success indicators
                                success_data = {k: v for k, v in extracted.items() if k.endswith('_success')}
                                if success_data:
                                    st.subheader("🎯 Field Success Indicators")
                                    success_df = pd.DataFrame([success_data])
                                    st.dataframe(success_df, hide_index=True)
                            else:
                                st.error(extracted['error'])
                        
                    except Exception as e:
                        st.error(f"Error previewing file: {str(e)}")
        else:
            st.info("Upload PDF files or a ZIP file first to see the preview")
            
            # Show what's currently uploaded
            if st.session_state.uploaded_files:
                st.write(f"✅ {len(st.session_state.uploaded_files)} PDF files uploaded")
            elif st.session_state.uploaded_zip:
                st.write(f"✅ ZIP file uploaded: {st.session_state.uploaded_zip.name}")
            else:
                st.write("❌ No files uploaded yet")
    
    with tab3:
        st.header("📊 Results Analysis")
        
        if 'results_df' in st.session_state:
            df = st.session_state['results_df']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Files Processed", len(df))
                
                # Check for errors
                error_count = df['error'].notna().sum() if 'error' in df.columns else 0
                st.metric("Files with Errors", error_count)
                
                # Count files that used offset method
                if 'extraction_method' in df.columns:
                    offset_count = (df['extraction_method'] == 'Y+45_offset').sum()
                    st.metric("Files Using Y+45 Offset", offset_count)
            
            with col2:
                # Basic statistics for numeric fields
                numeric_fields = ['property_coverage', 'loss_of_rent', 'total']
                for field in numeric_fields:
                    if field in df.columns:
                        # Convert to numeric, ignoring errors
                        numeric_values = pd.to_numeric(df[field], errors='coerce')
                        avg_value = numeric_values.mean()
                        if not pd.isna(avg_value):
                            st.metric(f"Average {field.replace('_', ' ').title()}", 
                                    f"${avg_value:,.2f}")
                            break
            
            with col3:
                # Show success rate metrics
                success_fields = [col for col in df.columns if col.endswith('_success')]
                if success_fields:
                    total_extractions = len(df) * len([col for col in success_fields])
                    successful_extractions = 0
                    for col in success_fields:
                        successful_extractions += (df[col] == '✅').sum()
                    
                    success_rate = (successful_extractions / total_extractions * 100) if total_extractions > 0 else 0
                    st.metric("Field Success Rate", f"{success_rate:.1f}%")
                else:
                    # Fallback to old method
                    main_fields = ['property_coverage', 'loss_of_rent', 'total', 'location_description']
                    total_fields = len(df) * len(main_fields)
                    empty_fields = 0
                    for field in main_fields:
                        if field in df.columns:
                            empty_fields += df[field].isna().sum() + (df[field] == '').sum()
                            empty_fields += df[field].str.contains('No text|OCR error|Invalid|failed', na=False).sum()
                    
                    completion_rate = ((total_fields - empty_fields) / total_fields * 100) if total_fields > 0 else 0
                    st.metric("Data Completion Rate", f"{completion_rate:.1f}%")
            
            # Show extraction method breakdown
            if 'extraction_method' in df.columns:
                st.subheader("🔧 Extraction Method Breakdown")
                method_counts = df['extraction_method'].value_counts()
                method_df = pd.DataFrame({
                    'Method': method_counts.index,
                    'Count': method_counts.values,
                    'Percentage': (method_counts.values / len(df) * 100).round(1)
                })
                st.dataframe(method_df, hide_index=True)
            
            # Show full results
            st.subheader("Full Results")
            st.dataframe(df, use_container_width=True)
            
            # Download section
            st.subheader("📥 Download Options")
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv_data,
                    file_name=f"insurance_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                try:
                    excel_buffer = io.BytesIO()
                    df.to_excel(excel_buffer, index=False, engine='openpyxl')
                    excel_data = excel_buffer.getvalue()
                    
                    st.download_button(
                        label="📊 Download as Excel",
                        data=excel_data,
                        file_name=f"insurance_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.error(f"Excel export not available: {e}")
            
        else:
            st.info("No results yet. Upload and process files in the Upload & Extract tab.")

if __name__ == "__main__":
    main()