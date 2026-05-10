from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from utils.document_utils import split_documents
from utils.knowledge_graph import SimpleKnowledgeGraph
from config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL,
    EMBEDDING_MODEL, VECTOR_DB_PATH, COLLECTION_NAME
)
import json

class ContentParsingAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            model=LLM_MODEL,
            temperature=0
        )
        self.embeddings = OpenAIEmbeddings(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            model=EMBEDDING_MODEL
        )
        self.vector_db = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME
        )
        self.knowledge_graph = SimpleKnowledgeGraph()
    
    def parse_and_index_documents(self, documents):
        splits = split_documents(documents)
        
        self.vector_db.add_documents(splits)
        
        for doc in splits:
            self._extract_entities_and_relations(doc.page_content, doc.metadata["source"])
        
        return len(splits)
    
    def _extract_entities_and_relations(self, text, source):
        prompt = f"""
        请从以下文本中提取关键实体和它们之间的关系。
        输出格式为JSON，包含两个字段：
        - entities: 实体列表，每个实体是一个对象，包含name和attributes(属性字典)
        - relations: 关系列表，每个关系是一个对象，包含entity1, relation, entity2
        
        文本内容：
        {text[:3000]}
        """
        
        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response.content)
            
            for entity in result.get("entities", []):
                self.knowledge_graph.add_entity(entity["name"], entity.get("attributes", {}))
            
            for rel in result.get("relations", []):
                self.knowledge_graph.add_relation(
                    rel["entity1"],
                    rel["relation"],
                    rel["entity2"],
                    source
                )
        except Exception as e:
            print(f"提取实体关系失败: {e}")
    
    def clear_index(self):
        self.vector_db.delete_collection()
        self.vector_db = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME
        )
        self.knowledge_graph.entities = {}
        self.knowledge_graph.relations = []
        self.knowledge_graph._save()
    
    def get_index_stats(self):
        return {
            "vector_count": self.vector_db._collection.count(),
            "entity_count": len(self.knowledge_graph.entities),
            "relation_count": len(self.knowledge_graph.relations)
        }