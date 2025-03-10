import streamlit as st
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from docx import Document as DocxDocument
from docx.shared import Pt
import tiktoken
import re

def count_tokens(text, model="cl100k_base"):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(text))

st.title("Document Summary Generator")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Add button after file upload
summarize_button = st.button("Summarize")

if summarize_button and uploaded_file is not None:
    try:
        with st.spinner("Processing document..."):
            pdf = PdfReader(uploaded_file)
            text = ''
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
            
            input_token_count = count_tokens(text)
            st.write(f"**Input Tokens:** {input_token_count}")
            
            docs = [Document(page_content=text)]
            
            llm = ChatGroq(
                groq_api_key='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', 
                model_name='llama3-70b-8192', 
                temperature=0.2, 
                top_p=0.2
            )
            
            template = '''Analyze the following document and extract:
            
            1. Overview (brief summary of the document in 3-5 lines)
            2. Involved Parties (key individuals or organizations mentioned)
            3. Issues before the Court (precise summary of what happened in 5 lines)
            4. Observation/Decision of the Court (precisely the important rulings or conclusions in 5 lines)
            
            Format your response exactly as:
            
            **Overview**
            · [Overview content]
            
            **Involved Parties**
            · [Names]
            
            **Issues before the Court**
            · [Summary of events]
            
            **Observation/Decision of the Court**
            · [Summary of findings]
            
            Here is the text to analyze:
            
            {text}
            '''
            
            prompt = PromptTemplate(input_variables=['text'], template=template)
            
            chain = load_summarize_chain(llm, chain_type='stuff', prompt=prompt, verbose=False)
            
            output_summary = chain.invoke(docs)
        
        output = output_summary['output_text']
        
        output_token_count = count_tokens(output)
        
        st.write(f"**Output Tokens:** {output_token_count}")
        st.write(f"**Total Tokens Used:** {input_token_count + output_token_count}")
        
        # Display the summary
        st.write("### Summary:")
        st.write(output)
        
        # Create DOCX document
        doc = DocxDocument()
        
        # Add document title
        title = doc.add_paragraph()
        title_run = title.add_run("Document Summary")
        title_run.bold = True
        title_run.font.size = Pt(16)
        
        # Add token usage information
        token_heading = doc.add_paragraph()
        token_heading_run = token_heading.add_run("Token Usage")
        token_heading_run.bold = True
        token_heading_run.font.size = Pt(14)
        
        token_info = doc.add_paragraph()
        token_info.add_run(f"Input Tokens: {input_token_count}\n").font.size = Pt(11)
        token_info.add_run(f"Output Tokens: {output_token_count}\n").font.size = Pt(11)
        token_info.add_run(f"Total Tokens: {input_token_count + output_token_count}").font.size = Pt(11)
        
        # Format and add the summary sections using the style from the first script
        sections = ["Overview", "Involved Parties", "Issues before the Court", "Observation/Decision of the Court"]
        
        for section in sections:
            section_pattern = rf'\*\*{section}\*\*\n(.*?)(?=\*\*|\Z)'
            section_match = re.search(section_pattern, output, re.DOTALL)
            
            if section_match:
                section_content = section_match.group(1).strip()
                
                # Add section heading
                section_heading = doc.add_paragraph()
                section_run = section_heading.add_run(section)
                section_run.bold = True
                section_run.font.size = Pt(14)
                
                # Split bullet points if they exist
                bullet_points = section_content.split('\n· ')
                
                for i, point in enumerate(bullet_points):
                    if i == 0 and not point.startswith('· '):
                        if point.startswith('·'):
                            point = point[1:].strip()
                        
                        bullet_para = doc.add_paragraph()
                        bullet_para.add_run('· ').font.size = Pt(11)
                        content_run = bullet_para.add_run(point.strip())
                        content_run.font.size = Pt(11)
                    elif i > 0:
                        bullet_para = doc.add_paragraph()
                        bullet_para.add_run('· ').font.size = Pt(11)
                        content_run = bullet_para.add_run(point.strip())
                        content_run.font.size = Pt(11)
        
        # Save and provide download
        doc_output_path = "document_summary.docx"
        doc.save(doc_output_path)
        
        with open(doc_output_path, "rb") as doc_file:
            st.download_button(
                "Download Summary DOCX",
                doc_file,
                file_name="document_summary.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please ensure you've uploaded a valid PDF file.")
else:
    st.info("Upload a PDF file and click 'Summarize' to process the document.")
