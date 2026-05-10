import os
import shutil
from utils.document_utils import load_documents_from_directory
from config import DOCUMENT_DIR

class DocumentCollectionAgent:
    def __init__(self):
        self.document_dir = DOCUMENT_DIR
    
    def upload_document(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        filename = os.path.basename(file_path)
        dest_path = os.path.join(self.document_dir, filename)
        
        if os.path.exists(dest_path):
            name, ext = os.path.splitext(filename)
            dest_path = os.path.join(self.document_dir, f"{name}_{int(os.path.getmtime(file_path))}{ext}")
        
        shutil.copy2(file_path, dest_path)
        return dest_path
    
    def scan_documents(self):
        return load_documents_from_directory(self.document_dir)
    
    def get_document_list(self):
        documents = []
        for root, _, files in os.walk(self.document_dir):
            for file in files:
                file_path = os.path.join(root, file)
                documents.append({
                    "filename": file,
                    "path": file_path,
                    "size": os.path.getsize(file_path),
                    "upload_time": os.path.getctime(file_path)
                })
        return documents
    
    def delete_document(self, file_path):
        if os.path.exists(file_path) and file_path.startswith(self.document_dir):
            os.remove(file_path)
            return True
        return False