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

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Add button after file upload
summarize_button = st.button("Summarize")

if summarize_button and uploaded_file is not None:
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
            
            llm = ChatGroq(
                groq_api_key='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', 
                model_name='llama3-70b-8192', 
                temperature=0.2, 
                top_p=0.2
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
            2. Involved Parties
            3. Issues before the Court (show the pointers, 4-5)
            4. Observation/Decision of the Court (display the pointers for important rulings or conclusions, 4-5)

            The following case notice follows a fixed format:
            - The Petitioner(s) name and address appear first, followed by the label "…Petitioner(s)"
            - Then the keyword "versus" or "vs" appears
            - After that, the Respondent(s) name and address are listed, ending with "…Respondent(s)"

            
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
                "Download Summary PDF",
                pdf_file,
                file_name=f"{filename_without_ext}_summary.pdf",
                mime="application/pdf"
            )
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please ensure you've uploaded a valid PDF file.")
else:
    st.info("Upload a PDF file and click 'Summarize' to process the document.")
