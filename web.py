import select
import gradio as gr
import os
import shutil
from models.localdocumentqa import LocalDocumentQA
from config.config import *


embedding_model_dict_list = list(embedding_model_dict.keys())

llm_model_dict_list = list(llm_model_dict.keys())

local_document_qa = LocalDocumentQA()


def get_file_list():
    if not os.path.exists(FILE_CONTENT_PATH):
        return []
    return [f for f in os.listdir(FILE_CONTENT_PATH)]    

file_list = get_file_list()

def upload_file(file):
    if not os.path.exists(FILE_CONTENT_PATH):
        os.mkdir(FILE_CONTENT_PATH)

    filename = os.path.basename(file.name)
    shutil.move(file.name,FILE_CONTENT_PATH + filename)

    file_list.insert(0, filename)
    return gr.Dropdown.update(choices=file_list,value=filename)



def get_answer(query,vector_store_path,history):


    # if vector_store_path:
    #     resp, history = local_document_qa.get_knowledge_based_answer(query=query,vector_store_path=vector_store_path,chat_history=history)
    # else:
    #     history = history + [[None,"请先加载文件后，再提问"]]

    for resp, history in local_document_qa.get_knowledge_based_answer(query=query, vector_store_path=vector_store_path, chat_history=history):
            source = "\n\n"
            source += "".join(
                [f"""<details> <summary>出处 [{i + 1}] {os.path.split(doc.metadata["source"])[-1]}</summary>\n"""
                 f"""{doc.page_content}\n"""
                 f"""</details>"""
                 for i, doc in
                 enumerate(resp["source_documents"])])
            history[-1][-1] += source
            yield history, ""

    # return history,""


def update_status(history, status):
    history = history + [[None,status]]
    print(status)
    return history


def init_model():

    try:
        local_document_qa.init_cfg()
        # local_document_qa.llm._call("你好") 
        return """ 模型已成功加载，请选择文件后点击"加载文件"按钮 """
    except Exception as e:
        print(e)
        return """ 模型加载异常，请选择文件后点击"加载文件"按钮 """

def reinit_model(llm_model,embedding_model,llm_history_len,top_k,history):

    try:
        local_document_qa.init_cfg(llm_model=llm_model,embedding_model=embedding_model,llm_history_len=llm_history_len,top_k=top_k)
        model_status = """ 模型已成功加载，请选择文件后点击"加载文件"按钮 """

    except:
        model_status = """ 模型加载异常，请选择文件后点击"加载文件"按钮 """

    return history + [[None,model_status]]


def get_vector_store(filepath,history):

    if local_document_qa.llm and local_document_qa.llm:
        vector_store_path = local_document_qa.init_knowledge_vector_store([FILE_CONTENT_PATH + filepath])

        if vector_store_path:
            file_status = "文件已经成功加载，请开始提问"
        else:
            file_status = "文件未加载成功，请重新上传"
    else:
        file_status = "模型未加载完成，请先加载模型后再导入文件"
        vector_store_path = None

    return vector_store_path,history + [[None,file_status]]

block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}

.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""


webui_title = """
# 🎉langchain-chatglm-document🎉

👍 [https://github.com/sunny5156/langchain-chatglm-document](https://github.com/sunny5156/langchain-chatglm-document)


"""

init_message = """
欢迎使用 langchain-chatglm-document Web UI，开始提问前，请依次如下 3 个步骤：
1. 选择语言模型、Embedding 模型及相关参数后点击"重新加载模型"，并等待加载完成提示
2. 上传或选择已有文件作为本地知识文档输入后点击"重新加载文档"，并等待加载完成提示
3. 输入要提交的问题后，点击回车提交"""

model_status = init_model()


if __name__ == "__main__":
    with gr.Blocks(css=block_css) as app:

        vector_store_path,file_status,model_status = gr.State(""),gr.State(""),gr.State(model_status)

        gr.Markdown(webui_title)

        with gr.Row():

            with gr.Column(scale=2):

                chatbot = gr.Chatbot([[None,init_message],[None,model_status.value]],elem_id="chat-box",show_label=False).style(height=750)

                query = gr.Textbox(show_label=False,placeholder="请输入提问内容，按回车键提交")


            with gr.Column(scale=1):
                llm_model = gr.Radio(llm_model_dict_list, label="LLM 模型",value=LLM_MODEL,interactive=True)

                llm_history_len = gr.Slider(0,10,value=3,step=1,label="LLM history len",interactive=True)

                embedding_model = gr.Radio(embedding_model_dict_list,label="Embedding 模型",value=EMBEDDING_MODEL,interactive=True)

                top_k = gr.Slider(1,20,value=6,step=1,label="向量匹配 topk",interactive=True)

                load_model_buttom = gr.Button("重新加载模型")

                with gr.Tab("select"):

                    selectFile = gr.Dropdown(file_list,label="content file",interactive=True,value=file_list[0] if len(file_list)>0 else None)

                with gr.Tab("upload"):
                    file = gr.File(label="content file",file_types=['.txt', '.md', '.docx', '.pdf'])
                
                load_file_button = gr.Button("加载文件")

        load_model_buttom.click(reinit_model,show_progress=True,inputs=[llm_model,embedding_model,llm_history_len,top_k,chatbot],outputs=chatbot)


        file.upload(upload_file,inputs=file,outputs=selectFile)

        load_file_button.click(get_vector_store,show_progress=True,inputs=[selectFile,chatbot],outputs=[vector_store_path,chatbot])

        query.submit(get_answer,inputs=[query,vector_store_path,chatbot],outputs=[chatbot,query])


    app.queue(concurrency_count=3).launch(server_name="0.0.0.0",share=False,inbrowser=False)
