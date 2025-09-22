import streamlit as st
try:
    import cv2
    # Test if cv2 is working properly
    _ = cv2.MORPH_OPENING
except (ImportError, AttributeError) as e:
    st.error(f"OpenCV issue detected: {e}")
    st.info("Using alternative image processing...")
    cv2 = None

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
    
    def scale_coordinates(self, coordinates, from_dpi, to_dpi):
        """Scale coordinates from one DPI to another"""
        if from_dpi == to_dpi:
            return coordinates
        
        scale_factor = to_dpi / from_dpi
        x1, y1, x2, y2 = coordinates
        return (
            int(x1 * scale_factor),
            int(y1 * scale_factor), 
            int(x2 * scale_factor),
            int(y2 * scale_factor)
        )
    
    def preprocess_image_pil(self, image_pil: Image.Image) -> Image.Image:
        """Preprocess image using PIL (fallback method)"""
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
            st.warning(f"PIL preprocessing failed: {e}")
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
        except Exception as e:
            st.warning(f"OpenCV preprocessing failed: {e}")
            return image
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy with fallback methods"""
        if cv2 is not None:
            try:
                return self.preprocess_image_cv2(image)
            except Exception as e:
                st.warning(f"OpenCV preprocessing failed, using PIL: {e}")
        
        # Fallback to PIL
        try:
            image_pil = Image.fromarray(image)
            processed_pil = self.preprocess_image_pil(image_pil)
            return np.array(processed_pil)
        except Exception as e:
            st.warning(f"All preprocessing failed: {e}")
            return image
    
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
            # For monetary amounts - more specific whitelist
            config = '--psm 8 -c tessedit_char_whitelist=0123456789.,$'
        elif 'location' in field_name.lower() or 'description' in field_name.lower():
            # For addresses - allow more characters
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
                # Remove everything except digits, decimal points, and commas
                text = re.sub(r'[^\d.,]', '', text)
                # Remove leading/trailing commas or dots
                text = text.strip('.,')
                # Handle multiple decimal points
                if text.count('.') > 1:
                    parts = text.split('.')
                    text = '.'.join(parts[:2])  # Keep only first decimal point
            elif 'location' in field_name.lower() or 'description' in field_name.lower():
                # For location/description - clean up but preserve address format
                # Remove excessive whitespace and line breaks
                text = ' '.join(text.split())
                # Remove common OCR artifacts
                text = re.sub(r'[|_\\]', '', text)
            
            return text if text else f"No text detected in {field_name}"
            
        except Exception as e:
            return f"OCR error in {field_name}: {str(e)}"
    
    def process_pdf(self, pdf_file) -> Dict[str, str]:
        """Process a single PDF file"""
        try:
            # Convert PDF to images at extraction DPI
            if hasattr(pdf_file, 'read'):
                # Streamlit uploaded file
                pdf_bytes = pdf_file.read()
                images = convert_from_bytes(pdf_bytes, dpi=self.extraction_dpi)
            else:
                # File path
                images = convert_from_path(pdf_file, dpi=self.extraction_dpi)
            
            # Process first page
            if not images:
                return {"error": "No pages found in PDF"}
            
            # Convert PIL image to numpy array
            image_np = np.array(images[0])
            
            # Extract all fields
            extracted_data = {}
            for field_name, coordinates in self.field_coordinates.items():
                value = self.extract_field_value(image_np, coordinates, field_name)
                extracted_data[field_name] = value
            
            return extracted_data
            
        except Exception as e:
            return {"error": f"Error processing PDF: {str(e)}"}
    
    def process_multiple_pdfs(self, pdf_files: List) -> pd.DataFrame:
        """Process multiple PDF files"""
        results = []
        progress_bar = st.progress(0)
        
        for i, pdf_file in enumerate(pdf_files):
            st.write(f"Processing: {pdf_file.name}")
            
            # Extract data from PDF
            extracted_data = self.process_pdf(pdf_file)
            
            # Add filename to results
            extracted_data['filename'] = pdf_file.name
            results.append(extracted_data)
            
            # Update progress
            progress_bar.progress((i + 1) / len(pdf_files))
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Reorder columns
        desired_order = ['filename', 'property_coverage', 'loss_of_rent', 'total', 'location_description']
        existing_cols = [col for col in desired_order if col in df.columns]
        other_cols = [col for col in df.columns if col not in desired_order]
        df = df[existing_cols + other_cols]
        
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
        **Based on your document:**
        - Property Coverage: Around (645, 1725, 1050, 1770)
        - Loss of Rent: Around (645, 1965, 1050, 2010)
        - Total: Around (530, 2320, 830, 2365)  
        - Location: Around (225, 1200, 1050, 1290)
        
        **Preview uses 200 DPI, extraction uses 300 DPI**
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
    """Safely draw rectangles with DPI scaling for preview"""
    # Scale coordinates from 300 DPI (extraction) to preview DPI
    scaled_coordinates = {}
    for field_name, coords in field_coordinates.items():
        x1, y1, x2, y2 = coords
        scale_factor = preview_dpi / 300  # Scale from 300 DPI to preview DPI
        scaled_coordinates[field_name] = (
            int(x1 * scale_factor),
            int(y1 * scale_factor),
            int(x2 * scale_factor), 
            int(y2 * scale_factor)
        )
    
    if cv2 is not None:
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
        except Exception as e:
            st.warning(f"OpenCV drawing failed: {e}")
    
    # Fallback: use PIL
    try:
        from PIL import ImageDraw, ImageFont
        
        image_pil = Image.fromarray(image_np)
        draw = ImageDraw.Draw(image_pil)
        
        colors = {
            'property_coverage': (255, 0, 0),      # Red
            'loss_of_rent': (0, 255, 0),          # Green  
            'total': (0, 0, 255),                 # Blue
            'location_description': (255, 255, 0)  # Yellow
        }
        
        for field_name, (x1, y1, x2, y2) in scaled_coordinates.items():
            color = colors.get(field_name, (255, 0, 0))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Add field label
            label = field_name.replace('_', ' ').title()
            draw.text((x1, y1-20), label, fill=color)
        
        return np.array(image_pil)
    except Exception as e:
        st.error(f"Both OpenCV and PIL drawing failed: {e}")
        return image_np

def main():
    st.title("📄 Insurance Document Data Extractor")
    st.markdown("Upload scanned PDF insurance documents to extract key information")
    
    # Show OpenCV status
    if cv2 is None:
        st.warning("⚠️ OpenCV not fully available. Using PIL fallback for image processing.")
    else:
        st.success("✅ OpenCV loaded successfully")
    
    # Get field coordinates from sidebar
    field_coordinates = get_field_coordinates_from_sidebar()
    
    # Additional sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Tips")
    st.sidebar.markdown("""
    - **Preview tab shows 200 DPI** (for speed)
    - **Extraction uses 300 DPI** (for accuracy)
    - Coordinates auto-scale between preview/extraction
    - Look for the colored boxes in preview
    - Property Coverage: Red box
    - Loss of Rent: Green box  
    - Total: Blue box
    - Location: Yellow box
    """)
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Extract", "🔍 Preview", "📊 Results"])
    
    with tab1:
        st.header("Upload PDF Files")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files with identical layouts"
        )
        
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} file(s)")
            
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
            if st.button("🚀 Extract Data", type="primary"):
                extractor = PDFExtractor(field_coordinates, extraction_dpi=300)
                
                with st.spinner("Processing PDFs..."):
                    results_df = extractor.process_multiple_pdfs(uploaded_files)
                
                # Store results in session state
                st.session_state['results_df'] = results_df
                st.success("✅ Extraction completed!")
                
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
        
        if uploaded_files:
            selected_file = st.selectbox("Select file to preview", uploaded_files, key="preview_file")
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.subheader("Field Colors")
                st.markdown("""
                - 🔴 **Property Coverage**
                - 🟢 **Loss of Rent** 
                - 🔵 **Total**
                - 🟡 **Location/Description**
                
                **Current 300 DPI Coordinates:**
                """)
                
                # Show scaled coordinates for 200 DPI preview
                st.text(f"Property: {field_coordinates['property_coverage']}")
                st.text(f"Loss Rent: {field_coordinates['loss_of_rent']}")  
                st.text(f"Total: {field_coordinates['total']}")
                st.text(f"Location: {field_coordinates['location_description']}")
            
            with col1:
                if selected_file and st.button("👁️ Show Extraction Areas"):
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
                            extracted = extractor.process_pdf(selected_file)
                            
                            if 'error' not in extracted:
                                preview_df = pd.DataFrame([extracted])
                                st.dataframe(preview_df, hide_index=True)
                            else:
                                st.error(extracted['error'])
                        
                    except Exception as e:
                        st.error(f"Error previewing file: {str(e)}")
        else:
            st.info("Upload a PDF file first to see the preview")
    
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
                # Show data quality metrics
                total_fields = len(df) * 4
                empty_fields = 0
                for field in ['property_coverage', 'loss_of_rent', 'total', 'location_description']:
                    if field in df.columns:
                        empty_fields += df[field].isna().sum() + (df[field] == '').sum()
                
                completion_rate = ((total_fields - empty_fields) / total_fields * 100) if total_fields > 0 else 0
                st.metric("Data Completion Rate", f"{completion_rate:.1f}%")
            
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