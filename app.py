import streamlit as st
import cv2
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes
import pandas as pd
import numpy as np
from PIL import Image
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

# Field coordinates (you'll need to calibrate these)
FIELD_COORDINATES = {
    'property_coverage': (646, 1717, 1363, 1769),  # Adjust these coordinates
    'loss_of_rent': (646, 1809, 1363, 1861),
    'total': (632, 2313, 1249, 2365),
    'location_description': (201, 1200, 918, 1294)
}

class PDFExtractor:
    def __init__(self):
        self.results = []
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPENING, kernel, iterations=1)
        
        return opening
    
    def extract_field_value(self, image: np.ndarray, coordinates: Tuple[int, int, int, int], field_name: str) -> str:
        """Extract text from specific coordinates"""
        x1, y1, x2, y2 = coordinates
        
        # Extract region of interest
        roi = image[y1:y2, x1:x2]
        
        # Preprocess ROI
        processed_roi = self.preprocess_image(roi)
        
        # Configure OCR based on field type
        if any(keyword in field_name.lower() for keyword in ['coverage', 'rent', 'total']):
            # For monetary amounts
            config = '--psm 7 -c tessedit_char_whitelist=0123456789.,'
        else:
            # For addresses and text
            config = '--psm 6'
        
        try:
            # Perform OCR
            text = pytesseract.image_to_string(processed_roi, config=config).strip()
            
            # Clean up the text
            if any(keyword in field_name.lower() for keyword in ['coverage', 'rent', 'total']):
                # Remove non-numeric characters except decimal points
                text = re.sub(r'[^\d.]', '', text)
                # Handle multiple decimal points
                parts = text.split('.')
                if len(parts) > 2:
                    text = parts[0] + '.' + ''.join(parts[1:])
            
            return text
        except Exception as e:
            st.error(f"Error extracting {field_name}: {str(e)}")
            return ""
    
    def process_pdf(self, pdf_file) -> Dict[str, str]:
        """Process a single PDF file"""
        try:
            # Convert PDF to images
            if hasattr(pdf_file, 'read'):
                # Streamlit uploaded file
                pdf_bytes = pdf_file.read()
                images = convert_from_bytes(pdf_bytes, dpi=300)
            else:
                # File path
                images = convert_from_path(pdf_file, dpi=300)
            
            # Process first page (assuming single page documents)
            if not images:
                return {"error": "No pages found in PDF"}
            
            # Convert PIL image to numpy array
            image_np = np.array(images[0])
            
            # Extract all fields
            extracted_data = {}
            for field_name, coordinates in FIELD_COORDINATES.items():
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

def main():
    st.title("📄 Insurance Document Data Extractor")
    st.markdown("Upload scanned PDF insurance documents to extract key information")
    
    # Sidebar for coordinate adjustment
    st.sidebar.title("⚙️ Configuration")
    
    with st.sidebar.expander("Adjust Field Coordinates"):
        st.write("Fine-tune extraction coordinates if needed:")
        
        # Property Coverage coordinates
        st.subheader("Property Coverage")
        pc_x1 = st.number_input("X1", value=450, key="pc_x1")
        pc_y1 = st.number_input("Y1", value=580, key="pc_y1")
        pc_x2 = st.number_input("X2", value=600, key="pc_x2")
        pc_y2 = st.number_input("Y2", value=610, key="pc_y2")
        FIELD_COORDINATES['property_coverage'] = (pc_x1, pc_y1, pc_x2, pc_y2)
        
        # Add similar controls for other fields...
    
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
            
            # Extract button
            if st.button("🚀 Extract Data", type="primary"):
                extractor = PDFExtractor()
                
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
        
        if uploaded_files:
            selected_file = st.selectbox("Select file to preview", uploaded_files)
            
            if selected_file and st.button("Show Extraction Areas"):
                try:
                    # Convert first page to image
                    pdf_bytes = selected_file.read()
                    images = convert_from_bytes(pdf_bytes, dpi=200)
                    
                    if images:
                        image_np = np.array(images[0])
                        
                        # Draw rectangles around extraction areas
                        for field_name, (x1, y1, x2, y2) in FIELD_COORDINATES.items():
                            cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(image_np, field_name, (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        
                        # Display image
                        st.image(image_np, caption="Extraction Areas (Red Rectangles)", 
                               use_column_width=True)
                        
                except Exception as e:
                    st.error(f"Error previewing file: {str(e)}")
    
    with tab3:
        st.header("📊 Results Analysis")
        
        if 'results_df' in st.session_state:
            df = st.session_state['results_df']
            
            col1, col2 = st.columns(2)
            
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
            
            # Show full results
            st.subheader("Full Results")
            st.dataframe(df, use_container_width=True)
            
        else:
            st.info("No results yet. Upload and process files in the Upload & Extract tab.")

if __name__ == "__main__":
    main()