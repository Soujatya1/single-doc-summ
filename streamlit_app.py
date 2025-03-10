import streamlit as st
import json
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from docx import Document as DocxDocument
from docx.shared import Pt
import tiktoken

def count_tokens(text, model="cl100k_base"):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(text))

st.title("Document Summary Generator")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    pdf = PdfReader(uploaded_file)
    text = ''
    for page in pdf.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    
    input_token_count = count_tokens(text)
    st.write(f"**Input Tokens:** {input_token_count}")
    
    docs = [Document(page_content=text)]
    
    llm = ChatGroq(groq_api_key='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', 
                   model_name='llama3-70b-8192', temperature=0.2, top_p=0.2)
    
    template = '''Write a very concise, well-explained, point-wise, short summary of the following text. Provide a structured response with the following sections:
    
    - Overview
    - Involved Parties
    - Issues before the Court
    - Observation/ Decision of the Court
    
    Text:
    '{text}'
    '''
    prompt = PromptTemplate(input_variables=['text'], template=template)
    
    chain = load_summarize_chain(llm, chain_type='stuff', prompt=prompt, verbose=False)
    
    with st.spinner("Generating summary..."):
        output_summary = chain.invoke(docs)
    
    output = output_summary['output_text']
    
    output_token_count = count_tokens(output)
    
    st.write(f"**Output Tokens:** {output_token_count}")
    st.write(f"**Total Tokens Used:** {input_token_count + output_token_count}")
    
    sections = ["Overview", "Involved Parties", "Key Events", "Key Findings"]
    summary_json = {}
    
    for section in sections:
        if section in output:
            start = output.find(section)
            next_section_index = float('inf')
            
            for s in sections:
                pos = output.find(s, start + len(section))
                if pos > -1 and pos < next_section_index:
                    next_section_index = pos
            
            if next_section_index == float('inf'):
                next_section_index = len(output)
                
            section_content = output[start + len(section):next_section_index].strip()
            summary_json[section] = section_content
    
    json_summary = json.dumps(summary_json, indent=4)
    
    st.write("### Summary (JSON format):")
    st.json(json_summary)
    
    json_output_path = "summary_output.json"
    with open(json_output_path, "w") as json_file:
        json_file.write(json_summary)
    
    with open(json_output_path, "rb") as json_file:
        st.download_button("Download Summary JSON", json_file, file_name="summary_output.json", mime="application/json")
    
    doc = DocxDocument()
    
    doc.add_paragraph("Token Usage", style='Heading 1')
    token_info = doc.add_paragraph()
    token_info.add_run(f"Input Tokens: {input_token_count}\n").font.size = Pt(11)
    token_info.add_run(f"Output Tokens: {output_token_count}\n").font.size = Pt(11)
    token_info.add_run(f"Total Tokens: {input_token_count + output_token_count}").font.size = Pt(11)
    
    if summary_json:
        for section, content in summary_json.items():
            doc.add_paragraph(section, style='Heading 1')
            paragraph = doc.add_paragraph(content)
            paragraph.runs[0].font.size = Pt(11)
    
    doc_output_path = "summary_output.docx"
    doc.save(doc_output_path)
    
    with open(doc_output_path, "rb") as doc_file:
        st.download_button("Download Summary DOCX", doc_file, file_name="summary_output.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
else:
    st.write("Please upload a PDF file.")
