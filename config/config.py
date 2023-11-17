
import torch.cuda


LLM_MODEL = "chatglm-6b"

LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

EMBEDDING_MODEL = "text2vec"

EMBEDDING_DEVICE = "cuda"

LLM_HISTORY_LEN = 3

VECTOR_SEARCH_TOP_K = 10

FILE_CONTENT_PATH = "data/content/"


embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
}

llm_model_dict = {
    "chatglm-6b-int4-qe": "THUDM/chatglm-6b-int4-qe",
    "chatglm-6b-int4": "THUDM/chatglm-6b-int4",
    "chatglm-6b": "THUDM/chatglm-6b",
}



