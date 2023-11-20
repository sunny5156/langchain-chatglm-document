from typing import Optional,List
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from config.config import LLM_DEVICE

from transformers import AutoTokenizer,AutoModel

import torch


DEVICE = LLM_DEVICE
DEVICE_ID = "0" if torch.cuda.is_available() else None
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()



class ChatGLM(LLM):

    max_token: int = 10000
    temperrature: float = 0.01
    top_p = 0.9
    history = []
    history_len = 10
    tokenizer: object = None
    model: object = None



    def __init__(self): 
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM3"


    def _call(self,prompt: str,stop:Optional[List[str]] = None) -> str:

        response, _=self.model.chat(self.tokenizer,prompt,history=self.history[-self.history_len:] if self.history_len > 0 else [],max_length=self.max_token,temperature=self.temperrature)

        torch_gc()

        if stop is not None:
            response = enforce_stop_tokens(response,stop)
        self.history = self.history+[[None,response]]


    def load_model(self,model_path: str = "THUDM/chatglm3-6b",llm_device=LLM_DEVICE):

        self.tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path,trust_remote_code=True).cuda()

        self.model = self.model.eval()
