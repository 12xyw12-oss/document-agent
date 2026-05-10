from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader,
    UnstructuredMarkdownLoader, CSVLoader, UnstructuredExcelLoader,
    UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from config import SUPPORTED_FORMATS, CHUNK_SIZE, CHUNK_OVERLAP

def load_single_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    
    loaders = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".csv": CSVLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".pptx": UnstructuredPowerPointLoader
    }
    
    if ext not in loaders:
        raise ValueError(f"不支持的文件格式: {ext}")
    
    loader = loaders[ext](file_path)
    return loader.load()

def load_documents_from_directory(directory):
    documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_FORMATS:
                file_path = os.path.join(root, file)
                try:
                    docs = load_single_document(file_path)
                    for doc in docs:
                        doc.metadata["source"] = file_path
                        doc.metadata["filename"] = file
                        doc.metadata["file_type"] = ext
                    documents.extend(docs)
                except Exception as e:
                    print(f"加载文档 {file_path} 失败: {e}")
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    return text_splitter.split_documents(documents)