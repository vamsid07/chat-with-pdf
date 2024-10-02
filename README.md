# PDF Chat Application

This application allows users to upload PDF documents, ask questions about their content, and receive answers based on the document's information. It also includes features like document summarization and text-to-speech capabilities.

## Features

1. **PDF Upload**: Users can upload PDF documents through the Streamlit interface.
2. **Document Embedding**: The application uses NVIDIA AI Endpoints for embedding the document content.
3. **Question Answering**: Users can ask questions about the uploaded document and receive relevant answers.
4. **Document Summarization**: The application can generate a concise summary of the uploaded document.
5. **Text-to-Speech**: Answers can be converted to speech using the Sarvam AI API.
6. **Chat History**: The application maintains a history of questions and answers for easy reference.

## Technologies Used

- **Python**: The primary programming language used for the application.
- **Streamlit**: Used for creating the web application interface.
- **LangChain**: Utilized for various NLP tasks, including document loading, text splitting, and chain creation.
- **NVIDIA AI Endpoints**: Used for embeddings and the language model (ChatNVIDIA).
- **FAISS**: Employed for efficient similarity search and clustering of dense vectors.
- **PyPDF DirectoryLoader**: Used for loading PDF documents.
- **Sarvam AI API**: Integrated for text-to-speech functionality.
- **dotenv**: Used for managing environment variables.

## Setup and Installation

1. Ensure you have Python installed on your system.
2. Clone this repository to your local machine.
3. Install the required dependencies:
   ```
   pip install streamlit langchain langchain_nvidia_ai_endpoints langchain_community python-dotenv faiss-cpu requests
   ```
4. Set up your environment variables:
   - Create a `.env` file in the project root.
   - Add your NVIDIA API key: `NVIDIA_API_KEY=your_api_key_here`
5. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. Launch the application using the command above.
2. Use the sidebar to upload a PDF document.
3. Click "Upload & Embed" to process the document.
4. Use the "Summarize Document" button to get a summary of the uploaded PDF.
5. Enter questions in the text input field and click "Ask" to get answers based on the document content.
6. Explore the chat history and document similarity search results in the expandable sections.

## Note

This application requires valid API keys for NVIDIA AI Endpoints and Sarvam AI. Ensure you have the necessary permissions and credits to use these services.
