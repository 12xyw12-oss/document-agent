import gradio as gr
from agents.collection_agent import DocumentCollectionAgent
from agents.parsing_agent import ContentParsingAgent
from agents.retrieval_agent import RetrievalQAAgent
from agents.maintenance_agent import MaintenanceAgent
import os

collection_agent = DocumentCollectionAgent()
parsing_agent = ContentParsingAgent()
retrieval_agent = RetrievalQAAgent()
maintenance_agent = MaintenanceAgent()

def upload_file(files):
    if not files:
        return "请选择要上传的文件"
    
    results = []
    for file in files:
        try:
            dest_path = collection_agent.upload_document(file.name)
            results.append(f"✅ 上传成功: {os.path.basename(dest_path)}")
        except Exception as e:
            results.append(f"❌ 上传失败: {os.path.basename(file.name)} - {str(e)}")
    
    return "\n".join(results)

def build_index():
    try:
        documents = collection_agent.scan_documents()
        if not documents:
            return "没有找到可索引的文档"
        
        parsing_agent.clear_index()
        count = parsing_agent.parse_and_index_documents(documents)
        stats = parsing_agent.get_index_stats()
        
        return f"""
        ✅ 索引构建完成
        - 处理文档数: {len(documents)}
        - 向量片段数: {stats['vector_count']}
        - 提取实体数: {stats['entity_count']}
        - 提取关系数: {stats['relation_count']}
        """
    except Exception as e:
        return f"❌ 索引构建失败: {str(e)}"

def answer_question(question):
    if not question:
        return "请输入您的问题"
    
    try:
        answer = retrieval_agent.answer_question(question)
        return answer
    except Exception as e:
        return f"❌ 回答失败: {str(e)}"

def incremental_sync():
    return maintenance_agent.incremental_sync()

def submit_feedback(question, answer, rating, comment):
    return maintenance_agent.add_feedback(question, answer, rating, comment)

with gr.Blocks(title="企业文档智能处理与知识沉淀Agent") as demo:
    gr.Markdown("# 企业文档智能处理与知识沉淀Agent")
    
    with gr.Tab("文档管理"):
        gr.Markdown("## 上传文档")
        file_upload = gr.File(file_count="multiple", label="选择文档")
        upload_btn = gr.Button("上传文档")
        upload_result = gr.Textbox(label="上传结果", lines=3)
        
        gr.Markdown("## 构建索引")
        build_btn = gr.Button("构建/重建索引")
        build_result = gr.Textbox(label="构建结果", lines=5)
        
        gr.Markdown("## 增量同步")
        sync_btn = gr.Button("同步新文档")
        sync_result = gr.Textbox(label="同步结果", lines=2)
    
    with gr.Tab("智能问答"):
        gr.Markdown("## 文档智能问答")
        question_input = gr.Textbox(label="请输入您的问题", placeholder="例如：项目X的截止日期是什么时候？")
        answer_btn = gr.Button("提问")
        answer_output = gr.Textbox(label="回答", lines=10)
        
        gr.Markdown("## 反馈")
        rating = gr.Slider(1, 5, value=3, step=1, label="回答质量评分")
        comment = gr.Textbox(label="补充意见", placeholder="可选")
        feedback_btn = gr.Button("提交反馈")
        feedback_result = gr.Textbox(label="反馈结果", lines=2)
    
    with gr.Tab("系统状态"):
        gr.Markdown("## 系统统计")
        stats_btn = gr.Button("刷新统计")
        stats_output = gr.Textbox(label="系统状态", lines=10)
    
    upload_btn.click(upload_file, inputs=[file_upload], outputs=[upload_result])
    build_btn.click(build_index, outputs=[build_result])
    sync_btn.click(incremental_sync, outputs=[sync_result])
    answer_btn.click(answer_question, inputs=[question_input], outputs=[answer_output])
    feedback_btn.click(submit_feedback, inputs=[question_input, answer_output, rating, comment], outputs=[feedback_result])
    stats_btn.click(lambda: f"""
    文档数: {len(collection_agent.get_document_list())}
    向量数: {parsing_agent.get_index_stats()['vector_count']}
    实体数: {parsing_agent.get_index_stats()['entity_count']}
    关系数: {parsing_agent.get_index_stats()['relation_count']}
    {maintenance_agent.get_feedback_stats()}
    """, outputs=[stats_output])

if __name__ == "__main__":
    demo.launch(share=True)