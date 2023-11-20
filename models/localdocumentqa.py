
import datetime
from genericpath import isdir
from typing import List
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from pkg_resources import VersionConflict
from sympy import content
from config.config import *
from models.chatglm import ChatGLM

from langchain.document_loaders import UnstructuredFileLoader

from langchain.vectorstores import FAISS

from langchain.prompts import PromptTemplate

from langchain.chains import RetrievalQA

import sentence_transformers

import os


class LocalDocumentQA:

    llm: object = None
    embeddings: object = None


    def init_cfg(self,
                 embedding_model: str = EMBEDDING_MODEL,
                 embedding_device = EMBEDDING_DEVICE,
                 llm_history_len: int = LLM_HISTORY_LEN,
                 llm_model: str = LLM_MODEL,
                 llm_device = LLM_DEVICE,
                 top_k=VECTOR_SEARCH_TOP_K
                 ):
        
        self.llm = ChatGLM()

        self.llm.load_model(model_path=llm_model_dict[llm_model],llm_device=llm_device)

        self.llm.history_len = llm_history_len

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model])

        self.embeddings.client = sentence_transformers.SentenceTransformer(self.embeddings.model_name,device=embedding_device)

        self.top_k = top_k



    def init_knowledge_vector_store(self, filepath: str or List[str]):

        docs = None

        if isinstance(filepath,str):

            if not os.path.exists(filepath):
                print("路径不存在")
                return None
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]

                try:
                    loader = UnstructuredFileLoader(filepath,mode="elements")
                    docs = loader.load()
                    print(f"{file} 已经加在完成")

                except:
                    print(f"{file} 加载出现异常")
                    return None
            elif os.path.isdir(filepath):
                docs = []
                for file in os.listdir(filepath):

                    fullfilepath = os.path.join(filepath, file)

                    try:
                        loader = UnstructuredFileLoader(fullfilepath,mode="elements")
                        docs += loader.load()
                        print(f"{file} 已经加在完成")
                    except:
                        print(f"{file} 加载出现异常")
        else:
            docs = []
            for file in filepath:
                try:

                    loader = UnstructuredFileLoader(file,mode="elements")
                    docs += loader.load()
                    print(f"{file} 已经加在完成")
                except:
                    print(f"{file} 加载出现异常")


        vector_store = FAISS.from_documents(docs,self.embeddings)

        vector_store_path = f"""./data/vector_store/{os.path.splitext(file)[0]}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"""

        vector_store.save_local(vector_store_path)

        return vector_store_path if len(docs)>0 else None



    def get_knowledge_based_answer(self,query,vector_store_path,chat_history=[]):

        prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
    如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。

    已知内容:
    {{content}}
    
    问题:
    {{question}} """
        
        prompt = PromptTemplate(input_variables=["content","question"],template=prompt_template)

        print(prompt.format(content="代码解释", questiont="新特性"))

        self.llm.history = chat_history

        vector_store = FAISS.load_local(vector_store_path,self.embeddings)

        knowledge_chain = RetrievalQA.from_llm(llm=self.llm,retriever=vector_store.as_retrieval(search_kwargs={"k": self.top_k}),prompt=prompt)

        knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(input_variables=["page_content"], template="{{page_content}}")

        knowledge_chain.return_source_documents = True


        result = knowledge_chain({"query": query})

        self.llm.history[-1][0] = query

        return result, self.llm.history

