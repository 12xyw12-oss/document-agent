from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from utils.knowledge_graph import SimpleKnowledgeGraph
from config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL,
    EMBEDDING_MODEL, VECTOR_DB_PATH, COLLECTION_NAME
)

class RetrievalQAAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            model=LLM_MODEL,
            temperature=0.3
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
        
        self.qa_chain = self._build_qa_chain()
    
    def _build_qa_chain(self):
        prompt_template = """
        请基于以下上下文回答用户的问题。如果上下文没有相关信息，请明确说明你不知道答案，不要编造内容。
        回答要简洁准确，引用相关文档来源。
        
        上下文：
        {context}
        
        问题：{question}
        
        回答：
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_db.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    
    def answer_question(self, question):
        if not question or not question.strip():
            return "请输入有效的问题"
        
        kg_info = self._search_knowledge_graph(question)
        
        try:
            result = self.qa_chain.invoke({"query": question})
        except (TypeError, ValueError):
            result = self.qa_chain.invoke(question)
        
        if isinstance(result, dict):
            answer = result.get("result", "")
            source_docs = result.get("source_documents", [])
        else:
            answer = str(result)
            source_docs = []
        
        if not answer.strip():
            answer = "未找到相关信息"
        
        if kg_info:
            answer += "\n\n知识图谱相关信息：\n" + kg_info
        
        if source_docs:
            sources = set()
            for doc in source_docs:
                if hasattr(doc, "metadata") and "source" in doc.metadata:
                    sources.add(doc.metadata["source"])
            if sources:
                answer += "\n\n参考来源：\n" + "\n".join(sorted(sources))
        
        return answer
    
    def _search_knowledge_graph(self, question):
        keywords = question.split()
        related_entities = set()
        
        for keyword in keywords:
            entities = self.knowledge_graph.search_entities(keyword)
            related_entities.update(entities)
        
        if not related_entities:
            return ""
        
        info = []
        for entity in related_entities:
            attrs = self.knowledge_graph.get_entity_attributes(entity)
            relations = self.knowledge_graph.get_related_entities(entity)
            
            entity_info = f"- {entity}"
            if attrs:
                entity_info += f": {', '.join([f'{k}={v}' for k, v in attrs.items()])}"
            
            if relations:
                entity_info += "\n  相关关系："
                for rel_entity, rel, src in relations:
                    entity_info += f"\n  - {entity} {rel} {rel_entity}"
            
            info.append(entity_info)
        
        return "\n".join(info)