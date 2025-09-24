import streamlit as st
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListItem, ListFlowable
import re
import os
import tempfile
from google.cloud import vision
import io
from PIL import Image
import pdf2image
import fitz

st.title("Case Laws Summarizer")

st.sidebar.header("API Configuration")

groq_api_key = st.sidebar.text_input(
    "Groq API Key", 
    type="password", 
    placeholder="Enter your Groq API key here"
)

azure_openai_key = st.sidebar.text_input(
    "Azure OpenAI API Key", 
    type="password", 
    placeholder="Enter your Azure OpenAI API key here"
)

azure_endpoint = st.sidebar.text_input(
    "Azure OpenAI Endpoint", 
    placeholder="https://your-resource.openai.azure.com/"
)

google_vision_api_key = st.sidebar.text_input(
    "Google Vision API Key",
    type="password",
    placeholder="Enter your Google Vision API key"
)

if google_vision_api_key:
    st.session_state['google_vision_api_key'] = google_vision_api_key

st.subheader("Model Configuration")
model_provider = st.selectbox(
    "Select AI Provider",
    ["Groq", "Azure OpenAI"]
)

if model_provider == "Groq":
    model_name = st.selectbox(
        "Select Groq Model",
        ["meta-llama/llama-4-scout-17b-16e-instruct"],
        index=0
    )
elif model_provider == "Azure OpenAI":
    deployment_name = st.text_input(
        "Deployment Name",
        placeholder="your-gpt-deployment-name",
        help="The name of your Azure OpenAI deployment"
    )

st.divider()

st.subheader("Text Extraction Options")
ocr_mode = st.selectbox(
    "Text Extraction Method",
    ["Auto-detect", "Standard PDF Reader", "OCR (for scanned documents)"],
    help="Choose how to extract text from your PDF"
)

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

summarize_button = st.button("Summarize", type="primary")

def setup_vision_client():
    try:
        api_key = st.session_state.get('google_vision_api_key')
        
        if not api_key:
            st.error("Please enter your Google Vision API key in the sidebar.")
            return None
            
        # Set up client with API key
        client_options = {"api_key": api_key}
        client = vision.ImageAnnotatorClient(client_options=client_options)
        return client
    except Exception as e:
        st.error(f"Error setting up Google Vision client: {str(e)}")
        return None

def pdf_to_images(pdf_path):
    try:
        pdf_document = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            
            img_data = pix.tobytes("png")
            
            images.append({
                "data": img_data,
                "page": page_num + 1
            })
        
        pdf_document.close()
        return images
        
    except Exception as e:
        st.error(f"Error converting PDF to images with PyMuPDF: {str(e)}")
        return []

def extract_text_with_vision(pdf_path):
    vision_client = setup_vision_client()
    if not vision_client:
        return []
    
    try:
        images = pdf_to_images(pdf_path)
        documents = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, img_info in enumerate(images):
            status_text.text(f"Processing page {img_info['page']} with OCR...")
            progress_bar.progress((i + 1) / len(images))
            
            image = vision.Image(content=img_info["data"])
            response = vision_client.text_detection(image=image)
            
            if response.error.message:
                st.error(f"Vision API error on page {img_info['page']}: {response.error.message}")
                continue
            
            texts = response.text_annotations
            if texts:
                extracted_text = texts[0].description
                
                doc = Document(
                    page_content=extracted_text,
                    metadata={
                        "source": pdf_path,
                        "page": img_info["page"],
                        "type": "scanned_text",
                        "extraction_method": "google_vision"
                    }
                )
                documents.append(doc)
        
        progress_bar.empty()
        status_text.empty()
        return documents
        
    except Exception as e:
        st.error(f"Error with Google Vision API: {str(e)}")
        return []

def extract_text_standard(uploaded_file):
    try:
        pdf = PdfReader(uploaded_file)
        text = ''
        total_pages = len(pdf.pages)
        
        for i, page in enumerate(pdf.pages):
            content = page.extract_text()
            if content:
                text += content + "\n"
        
        return [Document(page_content=text)]
    except Exception as e:
        st.error(f"Error extracting text with standard method: {str(e)}")
        return []

def is_scanned_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf = PdfReader(file)
            total_text = ""
            total_pages = len(pdf.pages)
            
            pages_to_check = min(5, total_pages)
            
            for i in range(pages_to_check):
                try:
                    page = pdf.pages[i]
                    page_text = page.extract_text().strip()
                    total_text += page_text
                    

                    if len(page_text.strip()) < 50:
                        continue
                        
                except Exception as e:
                    continue
        
        text_length = len(total_text.strip())
        words_count = len(total_text.split())
        
        avg_text_per_page = text_length / pages_to_check if pages_to_check > 0 else 0
        avg_words_per_page = words_count / pages_to_check if pages_to_check > 0 else 0
        
        # Thresholds for detection
        is_likely_scanned = (
            avg_text_per_page < 200 or
            avg_words_per_page < 30 or
            text_length < 100
        )
        
        return is_likely_scanned, text_length
        
    except Exception as e:
        st.warning(f"Could not analyze PDF structure: {str(e)}")
        return True, 0

def extract_text_auto_detect(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        is_scanned, text_length = is_scanned_pdf(tmp_path)
        
        if is_scanned:
            st.info(f"Detected scanned document (extracted text length: {text_length}). Using OCR...")
            documents = extract_text_with_vision(tmp_path)
        else:
            st.info(f"Detected standard PDF (extracted text length: {text_length}). Using standard extraction...")
            uploaded_file.seek(0)
            documents = extract_text_standard(uploaded_file)
        
        return documents
        
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

def validate_inputs():
    if model_provider == "Groq":
        if not groq_api_key:
            st.error("Please enter your Groq API key")
            return False
    elif model_provider == "Azure OpenAI":
        if not azure_openai_key:
            st.error("Please enter your Azure OpenAI API key")
            return False
        if not azure_endpoint:
            st.error("Please enter your Azure OpenAI endpoint")
            return False
        if not deployment_name:
            st.error("Please enter your Azure OpenAI deployment name")
            return False
    
    if not uploaded_file:
        st.error("Please upload a PDF file")
        return False
    
    if ocr_mode == "OCR (for scanned documents)" and not google_vision_api_key:
        st.error("Please enter your Google Vision API key to use OCR")
        return False
    
    return True

if summarize_button:
    if validate_inputs():
        try:
            with st.spinner("Processing document..."):
                original_filename = uploaded_file.name
                filename_without_ext = os.path.splitext(original_filename)[0]
                
                if ocr_mode == "Standard PDF Reader":
                    st.info("Using standard PDF text extraction...")
                    docs = extract_text_standard(uploaded_file)
                elif ocr_mode == "OCR (for scanned documents)":
                    st.info("Using OCR for text extraction...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    try:
                        docs = extract_text_with_vision(tmp_path)
                    finally:
                        os.unlink(tmp_path)
                else:
                    docs = extract_text_auto_detect(uploaded_file)
                
                if not docs or not any(doc.page_content.strip() for doc in docs):
                    st.error("No text could be extracted from the document. Please check if the file is valid or try OCR mode for scanned documents.")
                    st.stop()
                
                combined_text = "\n".join([doc.page_content for doc in docs if doc.page_content.strip()])
                docs = [Document(page_content=combined_text)]
                
                st.success(f"Successfully extracted {len(combined_text)} characters of text")
                
                if model_provider == "Groq":
                    llm = ChatGroq(
                        groq_api_key=groq_api_key, 
                        model_name=model_name, 
                        temperature=0.2, 
                        top_p=0.2
                    )
                elif model_provider == "Azure OpenAI":
                    from langchain_openai import AzureChatOpenAI
                    llm = AzureChatOpenAI(
                        azure_endpoint=azure_endpoint,
                        api_key=azure_openai_key,
                        azure_deployment=deployment_name,
                        api_version="2024-02-01",
                        temperature=0.2
                    )
                
                pii_instructions = """
                IMPORTANT: DO NOT include any personally identifiable information (PII) in your summary, including:
                - Bank account numbers
                - Credit card numbers
                - Social security numbers
                - Passport numbers
                - Personal mobile numbers
                
                If you encounter such information, DO NOT include it in your summary.
                """
                
                template = f'''
                Analyze the following document titled "{filename_without_ext}" and extract:
                
                1. Overview (brief summary of the document in 6-7 lines - single paragraph)
                2. Involved Parties (key individuals or organizations mentioned divided under Petitioner and Respondents)
                3. Issues before the Court (show the pointers, 4-5)
                4. Observation/Decision of the Court (display the pointers for important rulings or conclusions, 4-5)

                
                {pii_instructions}
                
                Format your response exactly as:
                
                **{filename_without_ext}**
                
                **Overview**
                · [Overview content]
                
                **Involved Parties**
                · [Names]
                
                **Issues before the Court**
                · [Summary of events]
                
                **Observation/Decision of the Court**
                · [Summary of findings]
                
                Here is the text to analyze:
                
                {{text}}
                '''
                
                prompt = PromptTemplate(input_variables=['text'], template=template)
                
                chain = load_summarize_chain(llm, chain_type='stuff', prompt=prompt, verbose=False)
                
                output_summary = chain.invoke(docs)
            
            output = output_summary['output_text']
            
            st.success("Summary generated successfully!")
            st.write("### Summary:")
            st.write(output)
            
            pdf_output_path = "document_summary.pdf"
            doc = SimpleDocTemplate(pdf_output_path, pagesize=letter)
            
            styles = getSampleStyleSheet()
            title_style = styles['Title']
            heading_style = styles['Heading2']
            
            bullet_style = ParagraphStyle(
                'BulletPoint',
                parent=styles['Normal'],
                leftIndent=20,
                firstLineIndent=0,
                spaceBefore=2,
                spaceAfter=2
            )
            
            content = []
            
            content.append(Paragraph(filename_without_ext, title_style))
            content.append(Spacer(1, 12))
            
            sections = ["Overview", "Involved Parties", "Issues before the Court", "Observation/Decision of the Court"]
            
            for section in sections:
                section_pattern = rf'\*\*{re.escape(section)}\*\*([\s\S]*?)(?=\*\*\w|\Z)'
                section_match = re.search(section_pattern, output)
                
                if section_match:
                    section_content = section_match.group(1).strip()
                    
                    content.append(Paragraph(section, heading_style))
                    content.append(Spacer(1, 6))
                    
                    normalized_content = section_content.replace('• ', '· ').replace('* ', '· ')
                    bullet_points = re.split(r'\n\s*·\s*|\s*·\s*', normalized_content)
                    bullet_points = [p.strip() for p in bullet_points if p.strip()]
                    
                    bullets = []
                    for point in bullet_points:
                        point = re.sub(r'^[·•*]\s*', '', point)
                        bullets.append(ListItem(Paragraph(point, bullet_style)))
                    
                    if bullets:
                        bullet_list = ListFlowable(
                            bullets,
                            bulletType='bullet',
                            start=None,
                            bulletFontName='Helvetica',
                            bulletFontSize=8,
                            leftIndent=20
                        )
                        content.append(bullet_list)
                    
                    content.append(Spacer(1, 12))
            
            doc.build(content)
            
            with open(pdf_output_path, "rb") as pdf_file:
                st.download_button(
                    "Download Summary PDF",
                    pdf_file,
                    file_name=f"{filename_without_ext}_summary.pdf",
                    mime="application/pdf"
                )
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please ensure you've uploaded a valid PDF file and provided correct API credentials.")
    else:
        st.warning("Please fill in all required fields before summarizing.")
