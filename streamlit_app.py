import streamlit as st
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from docx import Document as DocxDocument
from docx.shared import Pt
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
            
            # Include the filename in the prompt
            template = f'''
            Analyze the following document titled "{filename_without_ext}" and extract:
            
            1. Overview (brief summary of the document in 3-5 lines)
            2. Involved Parties (key individuals or organizations mentioned)
            3. Issues before the Court (precise summary of what happened in 5 lines)
            4. Observation/Decision of the Court (precisely the important rulings or conclusions in 5 lines)
            
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
        
        # Create DOCX document
        doc = DocxDocument()
        
        # Add document name as title (using the filename without extension)
        title = doc.add_paragraph()
        title_run = title.add_run(filename_without_ext)
        title_run.bold = True
        title_run.font.size = Pt(16)
        
        # Format and add the summary sections
        sections = ["Overview", "Involved Parties", "Issues before the Court", "Observation/Decision of the Court"]
        
        for section in sections:
            # More precise pattern matching that handles different formatting variations
            section_pattern = rf'\*\*{re.escape(section)}\*\*([\s\S]*?)(?=\*\*\w|\Z)'
            section_match = re.search(section_pattern, output)
            
            if section_match:
                section_content = section_match.group(1).strip()
                
                # Add section heading
                section_heading = doc.add_paragraph()
                section_run = section_heading.add_run(section)
                section_run.bold = True
                section_run.font.size = Pt(14)
                
                # Improved bullet point handling
                # First normalize bullet points to ensure consistent format
                normalized_content = section_content.replace('• ', '· ').replace('* ', '· ')
                # Split by bullet points, handling different possible formats
                bullet_points = re.split(r'\n\s*·\s*|\s*·\s*', normalized_content)
                bullet_points = [p.strip() for p in bullet_points if p.strip()]
                
                for point in bullet_points:
                    # Use proper bullet points with appropriate indentation
                    bullet_para = doc.add_paragraph(style='List Bullet')
                    bullet_para.style.font.size = Pt(11)
                    # Remove any existing bullet characters at the start of the content
                    point = re.sub(r'^[·•*]\s*', '', point)
                    content_run = bullet_para.add_run(point)
                    content_run.font.size = Pt(11)
        
        # Save and provide download
        doc_output_path = "document_summary.docx"
        doc.save(doc_output_path)
        
        with open(doc_output_path, "rb") as doc_file:
            st.download_button(
                "Download Summary DOCX",
                doc_file,
                file_name=f"{filename_without_ext}_summary.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please ensure you've uploaded a valid PDF file.")
else:
    st.info("Upload a PDF file and click 'Summarize' to process the document.")
