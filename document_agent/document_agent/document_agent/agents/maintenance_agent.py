import os
import time
import json
from agents.collection_agent import DocumentCollectionAgent
from agents.parsing_agent import ContentParsingAgent
from config import DOCUMENT_DIR

class MaintenanceAgent:
    def __init__(self):
        self.collection_agent = DocumentCollectionAgent()
        self.parsing_agent = ContentParsingAgent()
        self.last_sync_time = 0
        self.feedback_file = "./data/feedback.json"
        self._load_feedback()
    
    def _load_feedback(self):
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                self.feedback = json.load(f)
        else:
            self.feedback = []
    
    def _save_feedback(self):
        with open(self.feedback_file, "w", encoding="utf-8") as f:
            json.dump(self.feedback, f, ensure_ascii=False, indent=2)
    
    def incremental_sync(self):
        current_time = time.time()
        new_files = []
        
        for root, _, files in os.walk(DOCUMENT_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                mtime = os.path.getmtime(file_path)
                if mtime > self.last_sync_time:
                    new_files.append(file_path)
        
        if not new_files:
            return "没有新文档需要同步"
        
        from utils.document_utils import load_single_document
        new_documents = []
        for file_path in new_files:
            try:
                docs = load_single_document(file_path)
                for doc in docs:
                    doc.metadata["source"] = file_path
                    doc.metadata["filename"] = os.path.basename(file_path)
                    doc.metadata["file_type"] = os.path.splitext(file_path)[1].lower()
                new_documents.extend(docs)
            except Exception as e:
                print(f"同步文档 {file_path} 失败: {e}")
        
        if new_documents:
            self.parsing_agent.parse_and_index_documents(new_documents)
            self.last_sync_time = current_time
            return f"同步完成，新增 {len(new_documents)} 个文档片段"
        
        return "没有新文档需要同步"
    
    def add_feedback(self, question, answer, rating, comment=""):
        self.feedback.append({
            "question": question,
            "answer": answer,
            "rating": rating,
            "comment": comment,
            "timestamp": time.time()
        })
        self._save_feedback()
        return "感谢您的反馈，我们将持续优化模型"
    
    def get_feedback_stats(self):
        if not self.feedback:
            return "暂无反馈数据"
        
        total = len(self.feedback)
        avg_rating = sum(f["rating"] for f in self.feedback) / total
        
        return f"""
        反馈统计：
        - 总反馈数：{total}
        - 平均评分：{avg_rating:.2f}/5
        - 好评率：{sum(1 for f in self.feedback if f["rating"] >= 4)/total*100:.1f}%
        """
    
    def cleanup_expired_documents(self, days=365):
        current_time = time.time()
        expired_count = 0
        
        for root, _, files in os.walk(DOCUMENT_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                ctime = os.path.getctime(file_path)
                if (current_time - ctime) > days * 86400:
                    self.collection_agent.delete_document(file_path)
                    expired_count += 1
        
        return f"清理完成，删除了 {expired_count} 个过期文档"