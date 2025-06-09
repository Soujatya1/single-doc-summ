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

st.title("Document Summary Generator")

# API Key inputs
st.subheader("API Configuration")
col1, col2 = st.columns(2)

with col1:
    groq_api_key = st.text_input(
        "Groq API Key", 
        type="password", 
        placeholder="Enter your Groq API key here",
        help="Get your API key from https://console.groq.com/"
    )

with col2:
    azure_openai_key = st.text_input(
        "Azure OpenAI API Key", 
        type="password", 
        placeholder="Enter your Azure OpenAI API key here",
        help="Get your API key from Azure OpenAI service"
    )

# Azure OpenAI additional configuration
azure_endpoint = st.text_input(
    "Azure OpenAI Endpoint", 
    placeholder="https://your-resource.openai.azure.com/",
    help="Your Azure OpenAI service endpoint URL"
)

# Model selection
st.subheader("Model Configuration")
model_provider = st.selectbox(
    "Select AI Provider",
    ["Groq", "Azure OpenAI"],
    help="Choose which AI service to use for summarization"
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

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Add button after file upload
summarize_button = st.button("Summarize", type="primary")

# Validation function
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
    
    return True

if summarize_button:
    if validate_inputs():
        try:
            with st.spinner("Processing document..."):
                # Extract the filename without extension
                original_filename = uploaded_file.name
                filename_without_ext = os.path.splitext(original_filename)[0]
                
                pdf = PdfReader(uploaded_file)
                text = ''
                for page in pdf.pages:
                    content = page.extract_text()
                    if content:
                        text += content + "\n"
                
                docs = [Document(page_content=text)]
                
                # Initialize LLM based on selected provider
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
                
                # PII instructions to integrate in the prompt
                pii_instructions = """
                IMPORTANT: DO NOT include any personally identifiable information (PII) in your summary, including:
                - Bank account numbers
                - Credit card numbers
                - Social security numbers
                - Passport numbers
                - Personal mobile numbers
                
                If you encounter such information, DO NOT include it in your summary.
                """
                
                # Include the filename and PII instructions in the prompt
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
            
            # Display the summary
            st.success("Summary generated successfully!")
            st.write("### Summary:")
            st.write(output)
            
            # Create PDF document
            pdf_output_path = "document_summary.pdf"
            doc = SimpleDocTemplate(pdf_output_path, pagesize=letter)
            
            # Set up styles
            styles = getSampleStyleSheet()
            title_style = styles['Title']
            heading_style = styles['Heading2']
            
            # Create bullet point style
            bullet_style = ParagraphStyle(
                'BulletPoint',
                parent=styles['Normal'],
                leftIndent=20,
                firstLineIndent=0,
                spaceBefore=2,
                spaceAfter=2
            )
            
            # Build the PDF content
            content = []
            
            # Add document name as title
            content.append(Paragraph(filename_without_ext, title_style))
            content.append(Spacer(1, 12))
            
            # Format and add the summary sections
            sections = ["Overview", "Involved Parties", "Issues before the Court", "Observation/Decision of the Court"]
            
            for section in sections:
                # More precise pattern matching that handles different formatting variations
                section_pattern = rf'\*\*{re.escape(section)}\*\*([\s\S]*?)(?=\*\*\w|\Z)'
                section_match = re.search(section_pattern, output)
                
                if section_match:
                    section_content = section_match.group(1).strip()
                    
                    # Add section heading
                    content.append(Paragraph(section, heading_style))
                    content.append(Spacer(1, 6))
                    
                    # Improved bullet point handling
                    # First normalize bullet points to ensure consistent format
                    normalized_content = section_content.replace('• ', '· ').replace('* ', '· ')
                    # Split by bullet points, handling different possible formats
                    bullet_points = re.split(r'\n\s*·\s*|\s*·\s*', normalized_content)
                    bullet_points = [p.strip() for p in bullet_points if p.strip()]
                    
                    # Create a list of bullet points
                    bullets = []
                    for point in bullet_points:
                        # Remove any existing bullet characters at the start of the content
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
            
            # Build the PDF
            doc.build(content)
            
            # Provide download button
            with open(pdf_output_path, "rb") as pdf_file:
                st.download_button(
                    "📄 Download Summary PDF",
                    pdf_file,
                    file_name=f"{filename_without_ext}_summary.pdf",
                    mime="application/pdf"
                )
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please ensure you've uploaded a valid PDF file and provided correct API credentials.")
    else:
        st.warning("Please fill in all required fields before summarizing.")
