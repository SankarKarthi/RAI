import streamlit as st
import fitz  
from transformers import pipeline, BertTokenizer, BertForQuestionAnswering
import torch

def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

st.title("PDF Question Answering App")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.text("PDF text extracted.")

    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    question = st.text_input("Type your question here")

    if st.button("Submit"):
        if question:
            inputs = tokenizer(question, pdf_text, return_tensors='pt', max_length=512, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            
            start_idx = torch.argmax(outputs.start_logits)
            end_idx = torch.argmax(outputs.end_logits) + 1
            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_idx:end_idx]))

            st.write(f"**Answer:** {answer}")
        else:
            st.write("Please enter a question.")
