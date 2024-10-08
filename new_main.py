import streamlit as st
import os
import requests
from io import BytesIO
from langchain.chains import LLMChain
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import json
import base64

load_dotenv()

os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

def vector_embedding(uploaded_file=None):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        if uploaded_file:
            with open(os.path.join("pdfs", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.loader = PyPDFDirectoryLoader("pdfs")
        else:
            st.session_state.loader = PyPDFDirectoryLoader("pdfs")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

def text_to_speech(text, target_language_code):
    try:
        url = "https://api.sarvam.ai/text-to-speech"
        payload = {
            "inputs": [text],
            "target_language_code": target_language_code,
            "speaker": "arvind",  # This might need to be adjusted based on language
            "pitch": 0,
            "pace": 1,
            "loudness": 1.5,
            "speech_sample_rate": 16000,
            "enable_preprocessing": True,
            "model": "bulbul:v1"
        }
        headers = {
            "api-subscription-key": "339d2447-2db9-4172-8ce6-17ebd38a790b",
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        json_data = json.loads(response.content)
        print(json_data)
        audio_base64 = json_data['audios'][0]
        audio_data = base64.b64decode(audio_base64)

        return BytesIO(audio_data)

    except requests.exceptions.RequestException as e:
        st.error(f"Error in API request: {str(e)}")
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON response: {str(e)}")
    except KeyError as e:
        st.error(f"Error accessing audio data from response: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
    
    return None


def translate_text(text, target_language_code):
    url = "https://api.sarvam.ai/translate"
    payload = {
        "input": text,
        "source_language_code": "en-IN",
        "target_language_code": target_language_code,
        "speaker_gender": "Male",
        "mode": "formal",
        "model": "mayura:v1",
        "enable_preprocessing": True
    }
    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": "339d2447-2db9-4172-8ce6-17ebd38a790b"
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["translated_text"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error in translation API request: {str(e)}")
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON response from translation API: {str(e)}")
    except KeyError as e:
        st.error(f"Error accessing translated text from response: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error during translation: {str(e)}")
    return None

def is_greeting(text):
    greetings = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
    return any(greeting in text.lower() for greeting in greetings)

def handle_greeting(text):
    default_greeting = "Hello! Welcome to the PDF Chat. How can I assist you today?"
    return f"{default_greeting}"

def summarize_document():
    if "final_documents" not in st.session_state or not st.session_state.final_documents:
        return "Please upload a document first."
    
    try:
        full_text = " ".join([doc.page_content for doc in st.session_state.final_documents])
        
        summary_prompt = ChatPromptTemplate.from_template(
            """
            Please provide a concise summary of the following document:
            {text}
            
            Summary:
            """
        )
        
        summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
        summary = summary_chain.run(text=full_text[:4000])
        
        return summary
    except ValueError as e:
        st.error(f"Error during summarization: {str(e)}")
        return "An error occurred while summarizing the document. Please try again."
    except Exception as e:
        st.error(f"Unexpected error during summarization: {str(e)}")
        return "An unexpected error occurred. Please try again or contact support."

st.title("Chat with PDF")

st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_file and st.sidebar.button("Upload & Embed"):
    vector_embedding(uploaded_file)
    st.sidebar.success("Document uploaded and embedded!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.sidebar.button("Summarize Document"):
    with st.spinner("Generating summary..."):
        summary = summarize_document()
        st.write("### Document Summary")
        st.write(summary)

prompt1 = st.text_input("Ask a question from the document")

language_options = {
    "English": "en-IN",
    "Tamil": "ta-IN",
    "Hindi": "hi-IN",
    "Telugu": "te-IN",
    "Kannada": "kn-IN"
}

selected_translation_language = st.selectbox("Select language for translation", list(language_options.keys()))

selected_tts_language = st.selectbox("Select language for text-to-speech", list(language_options.keys()))

if st.button("Ask"):
    if prompt1:
        if is_greeting(prompt1):
            answer = handle_greeting(prompt1)
        else:
            if "vectors" not in st.session_state:
                vector_embedding()

            prompt = ChatPromptTemplate.from_template(
                """
                Answer the questions based on the provided context only.
                Please provide the most accurate response based on the question:
                <context>
                {context}
                </context>
                Questions: {input}
                """
            )
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            response = retrieval_chain.invoke({'input': prompt1})
            answer = response['answer']

        st.session_state.chat_history.append({"question": prompt1, "answer": answer})

        st.write("### Answer (English)")
        st.write(answer)

        if selected_translation_language != "English":
            translated_answer = translate_text(answer, language_options[selected_translation_language])
            if translated_answer:
                st.write(f"### Translated Answer ({selected_translation_language})")
                st.write(translated_answer)
            else:
                st.error("Translation failed. Displaying original English answer.")
    
        tts_text = translated_answer if selected_translation_language != "English" and translated_answer else answer
        audio_data = text_to_speech(tts_text, language_options[selected_tts_language])
        if audio_data:
            st.write(f"### Audio ({selected_tts_language})")
            st.audio(audio_data, format='audio/wav')
        else:
            st.error("Text-to-speech conversion failed.")

        with st.expander("Chat History"):
            for i, chat in enumerate(st.session_state.chat_history):
                st.write(f"**Q{i+1}:** {chat['question']}")
                st.write(f"**A{i+1}:** {chat['answer']}")
                st.write("-----------------------------------")

        if not is_greeting(prompt1):
            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(response.get("context", [])):
                    st.write(doc.page_content)
                    st.write("-----------------------------------")