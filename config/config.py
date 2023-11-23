
import torch.cuda


LLM_MODEL = "chatglm2-6b-int4"

LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

EMBEDDING_MODEL = "text2vec"

EMBEDDING_DEVICE = "cuda"

LLM_HISTORY_LEN = 3

VECTOR_SEARCH_TOP_K = 10

CHUNK_SIZE = 250

FILE_CONTENT_PATH = "data/content/"

# 知识检索内容相关度 Score, 数值范围约为0-1100，如果为0，则不生效，经测试设置为小于500时，匹配结果更精准
VECTOR_SEARCH_SCORE_THRESHOLD = 0


embedding_model_dict = {
    # "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    # "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "/F/aistudio/super/text2vec-large-chinese",
}

llm_model_dict = {
    # "chatglm-6b-int4-qe": "THUDM/chatglm-6b-int4-qe",
    # "chatglm-6b-int4": "THUDM/chatglm-6b-int4",
    "chatglm3-6b": "/F/aistudio/super/chatglm3-6b",
    "chatglm2-6b-int4": "/F/aistudio/super/chatglm2-6b-int4",
}



