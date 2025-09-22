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

# Default field coordinates (you'll need to calibrate these)
DEFAULT_COORDINATES = {
    'property_coverage': (430, 1150, 700, 1180),
    'loss_of_rent': (430, 1280, 700, 1238),
    'total': (353, 1547, 653, 1577),
    'location_description': (150, 800, 700, 870)
}

class PDFExtractor:
    def __init__(self, field_coordinates):
        self.results = []
        self.field_coordinates = field_coordinates
    
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
    st.sidebar.markdown("Adjust extraction coordinates for each field:")
    
    # Property Coverage
    with st.sidebar.expander("📊 Property Coverage", expanded=False):
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
    
    # Save coordinates button
    if st.sidebar.button("💾 Save Current Coordinates"):
        st.sidebar.success("Coordinates saved for this session!")
        # You could add code here to save to a file or database
    
    return coordinates

def main():
    st.title("📄 Insurance Document Data Extractor")
    st.markdown("Upload scanned PDF insurance documents to extract key information")
    
    # Get field coordinates from sidebar
    field_coordinates = get_field_coordinates_from_sidebar()
    
    # Additional sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Tips")
    st.sidebar.markdown("""
    - Use the **Preview** tab to see extraction areas
    - Adjust coordinates if red boxes don't align with text
    - X1,Y1 = top-left corner
    - X2,Y2 = bottom-right corner
    - Add 5-10 pixel buffer around text
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
            with st.expander("📐 Current Coordinates", expanded=False):
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
                extractor = PDFExtractor(field_coordinates)
                
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
            selected_file = st.selectbox("Select file to preview", uploaded_files, key="preview_file")
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.subheader("Field Colors")
                st.markdown("""
                - 🔴 **Property Coverage**
                - 🟢 **Loss of Rent** 
                - 🔵 **Total**
                - 🟡 **Location/Description**
                """)
            
            with col1:
                if selected_file and st.button("👁️ Show Extraction Areas"):
                    try:
                        # Convert first page to image
                        pdf_bytes = selected_file.read()
                        images = convert_from_bytes(pdf_bytes, dpi=200)
                        
                        if images:
                            image_np = np.array(images[0])
                            
                            # Define colors for each field
                            colors = {
                                'property_coverage': (255, 0, 0),      # Red
                                'loss_of_rent': (0, 255, 0),          # Green
                                'total': (0, 0, 255),                 # Blue
                                'location_description': (255, 255, 0)  # Yellow
                            }
                            
                            # Draw rectangles around extraction areas
                            for field_name, (x1, y1, x2, y2) in field_coordinates.items():
                                color = colors.get(field_name, (255, 0, 0))
                                cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 3)
                                
                                # Add field label
                                label = field_name.replace('_', ' ').title()
                                cv2.putText(image_np, label, (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            
                            # Display image
                            st.image(image_np, caption="Extraction Areas", use_column_width=True)
                            
                            # Show extracted values
                            st.subheader("🔍 Extracted Values Preview")
                            extractor = PDFExtractor(field_coordinates)
                            extracted = extractor.process_pdf(selected_file)
                            
                            if 'error' not in extracted:
                                preview_df = pd.DataFrame([extracted])
                                st.dataframe(preview_df, hide_index=True)
                            else:
                                st.error(extracted['error'])
                        
                    except Exception as e:
                        st.error(f"Error previewing file: {str(e)}")
    
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
                            break  # Show only one metric to fit space
            
            with col3:
                # Show data quality metrics
                total_fields = len(df) * 4  # 4 fields per document
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
                excel_buffer = io.BytesIO()
                df.to_excel(excel_buffer, index=False, engine='openpyxl')
                excel_data = excel_buffer.getvalue()
                
                st.download_button(
                    label="📊 Download as Excel",
                    data=excel_data,
                    file_name=f"insurance_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
        else:
            st.info("No results yet. Upload and process files in the Upload & Extract tab.")

if __name__ == "__main__":
    main()