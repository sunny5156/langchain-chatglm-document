
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

from models.chinese_text_splitter import ChineseTextSplitter
from utils import torch_gc

from models.myfiass import MyFAISS
from langchain.docstore.document import Document


def generate_prompt(related_docs: List[str],
                    query: str,
                    prompt_template: str = "", ) -> str:
    context = "\n".join([doc.page_content for doc in related_docs])
    prompt = prompt_template.replace("{question}", query).replace("{context}", context)
    return prompt

def search_result2docs(search_results):
    """
    将bing搜索返回的结果封装为Document类实例
    """
    docs = []
    for result in search_results:
        doc = Document(page_content=result["snippet"] if "snippet" in result.keys() else "",
                       metadata={"source": result["link"] if "link" in result.keys() else "",
                                 "filename": result["title"] if "title" in result.keys() else ""})
        docs.append(doc)
    return docs

class LocalDocumentQA:

    llm: object = None
    embeddings: object = None
    chunk_size: int = CHUNK_SIZE
    chunk_conent: bool = True


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

    def load_file(self, filepath):
        if filepath.lower().endswith(".md"):
            loader = UnstructuredFileLoader(filepath, mode="elements")
            docs = loader.load()
        elif filepath.lower().endswith(".pdf"):
            loader = UnstructuredFileLoader(filepath)
            textsplitter = ChineseTextSplitter(pdf=True)
            docs = loader.load_and_split(textsplitter)
        else:
            loader = UnstructuredFileLoader(filepath, mode="elements")
            textsplitter = ChineseTextSplitter(pdf=False)
            docs = loader.load_and_split(text_splitter=textsplitter)
        return docs

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
                    # docs = self.load_file(filepath)
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
                        # docs += self.load_file(fullfilepath)
                        print(f"{file} 已经加在完成")
                    except Exception as e:
                        # print(e)
                        print(f"{file} 加载出现异常")
        else:
            docs = []
            for file in filepath:
                try:

                    loader = UnstructuredFileLoader(file,mode="elements")
                    docs += loader.load()
                    # docs += self.load_file(file)
                    print(f"{file} 已经加在完成")
                except Exception as e:
                    # print(e)
                    print(f"{file} 加载出现异常")

        # vector_store = FAISS.from_documents(docs,self.embeddings)
        vector_store = MyFAISS.from_documents(docs, self.embeddings)  # docs 为Document列表

        vector_store_path = f"""./data/vector_store/{os.path.splitext(file)[0]}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"""
        vector_store.save_local(vector_store_path)

        return vector_store_path if len(docs)>0 else None

    def get_docs_with_score(docs_with_score):
        docs = []
        for doc, score in docs_with_score:
            doc.metadata["score"] = score
            docs.append(doc)
        return docs


    def get_knowledge_based_answer(self,query,vector_store_path,chat_history=[]):

        prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
    如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。

    已知内容:
    {content}
    
    问题:
    {question} """
        
        prompt = PromptTemplate(input_variables=["content","question"],template=prompt_template)

        self.llm.history = chat_history

        # vector_store = FAISS.load_local(vector_store_path,self.embeddings)
        # knowledge_chain = RetrievalQA.from_llm(llm=self.llm,retriever=vector_store.as_retriever(search_kwargs={"k": self.top_k}),prompt=prompt)
        # knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(input_variables=["page_content"], template="{page_content}")
        # knowledge_chain.return_source_documents = True
        # result = knowledge_chain({"query": query})

        ###langchain 0.0.174 
        # 模型载入
        vector_store = MyFAISS.load_local(vector_store_path, self.embeddings)
        # FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        vector_store.chunk_size = self.chunk_size
        # 利用faiss求相似
        related_docs_with_score = vector_store.similarity_search_with_score(query, k=self.top_k)
        # 单独抽取出文档
        # related_docs = self.get_docs_with_score(related_docs_with_score)
        torch_gc()

        if len(related_docs_with_score) > 0:
            prompt = generate_prompt(related_docs_with_score, query)
        else:
            prompt = query

        for answer_result in self.llm.generatorAnswer(prompt=prompt, history=chat_history, streaming=False):
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][0] = query
            response = {"query": query,
                        "result": resp,
                        "source_documents": related_docs_with_score}
            yield response, history

