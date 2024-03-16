from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

import langchain
langchain.verbose = False

# Function to process text from pdf
def process_text(text, openai_api_key):
    # Split the text into chunks using langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base


def main():
    st.title("Scoremint Chat Bot 🤖")
    st.write("Simply Upload Your Marksheet And Put the OpenAI API Key to start retrieving information the Cool Way 😉.")

    # Input for OpenAI API key
    openai_api_key = st.sidebar.text_input('Enter your OpenAI API Key', key="openai_api_key")

    # Check if OpenAI API key is provided and valid
    if openai_api_key:
        pdf = st.file_uploader("Upload your Marksheet (PDF)", type="pdf")

        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            
            # Store the pdf text in a variable
            text = ""

            for page in pdf_reader.pages:
                text += page.extract_text()

            # Create a knowledge base object
            knowledgeBase = process_text(text, openai_api_key)

            query = st.text_input('Ask question about the Marksheet...', key="query")

            cancel_button = st.button('Cancel')

            if cancel_button:
                st.stop()

            if query:
                docs = knowledgeBase.similarity_search(query)

                llm = OpenAI(openai_api_key=openai_api_key)

                chain = load_qa_chain(llm, chain_type="stuff")

                with get_openai_callback() as cost:
                    response = chain.invoke(input={"question": query, "input_documents": docs})
                    print(cost)

                    st.write(response["output_text"])
    else:
        st.warning("Please enter your OpenAI API key to proceed.")


if __name__ == "__main__":
    main()
