import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import DB_DIR, APIConfig
from src.logger import get_logger

logger = get_logger(__name__)

# 使用配置檔案中的路徑
PERSIST_DIRECTORY = str(DB_DIR)
PDF_DIRECTORY = APIConfig.PDF_DIRECTORY

def build_knowledge_base():
    """
    Builds the knowledge base from PDF documents and stores it in a Chroma vector store.
    """
    documents = []
    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in {PDF_DIRECTORY}. Please ensure your PDF documents are in the root directory.")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIRECTORY, pdf_file)
        print(f"Loading PDF: {pdf_path}")
        try:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")

    if not documents:
        print("No documents were loaded. Exiting knowledge base creation.")
        return

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=APIConfig.CHUNK_SIZE,
        chunk_overlap=APIConfig.CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)
    logger.info(f"將 {len(documents)} 份文件分割成 {len(texts)} 個片段")
    print(f"Split {len(documents)} documents into {len(texts)} chunks.")

    # Create embeddings using a HuggingFace model
    logger.info("建立嵌入向量...")
    print("Creating embeddings (this may take a while)...")
    embeddings = HuggingFaceEmbeddings(model_name=APIConfig.EMBEDDING_MODEL_NAME)

    # Create and persist the Chroma vector store
    print(f"Creating Chroma vector store in {PERSIST_DIRECTORY}...")
    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=PERSIST_DIRECTORY)
    vectordb.persist()
    print(f"Knowledge base built and saved to {PERSIST_DIRECTORY}.")

if __name__ == "__main__":
    build_knowledge_base()
