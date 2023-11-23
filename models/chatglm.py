import re
from typing import Optional,List
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from config.config import LLM_DEVICE

from transformers import AutoTokenizer,AutoModel

import torch

from utils import torch_gc
from models.base import (BaseAnswer, AnswerResult)


DEVICE = LLM_DEVICE
DEVICE_ID = "0" if torch.cuda.is_available() else None
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


class ChatGLM(BaseAnswer,LLM):

    max_token: int = 10000
    temperature: float = 0.01
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
    

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len


    def _call(self,prompt: str,stop:Optional[List[str]] = None) -> str:

        response, _=self.model.chat(self.tokenizer,
                                    prompt,
                                    history=self.history[-self.history_len:] if self.history_len > 0 else [],
                                    max_length=self.max_token,
                                    temperature=self.temperature
                                    )

        torch_gc()

        if stop is not None:
            response = enforce_stop_tokens(response,stop)
        self.history = self.history+[[None,response]]
        return response


    def load_model(self,model_path: str = "THUDM/chatglm3-6b",llm_device=LLM_DEVICE):

        self.tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path,trust_remote_code=True).half().cuda()

        self.model = self.model.eval()

    
    def generatorAnswer(self, prompt: str,
                         history: List[List[str]] = [],
                         streaming: bool = False):

        if streaming:
            history += [[]]
            for inum, (stream_resp, _) in enumerate(self.model.stream_chat(
                    self.tokenizer,
                    prompt,
                    history=history[-self.history_len:-1] if self.history_len > 1 else [],
                    max_length=self.max_token,
                    temperature=self.temperature
            )):
                history[-1] = [prompt, stream_resp]
                answer_result = AnswerResult()
                answer_result.history = history
                answer_result.llm_output = {"answer": stream_resp}
                yield answer_result
        else:
            response, _ = self.model.chat(
                self.tokenizer,
                prompt,
                history=history[-self.history_len:] if self.history_len > 0 else [],
                max_length=self.max_token,
                temperature=self.temperature
            )
            torch_gc()
            history += [[prompt, response]]
            answer_result = AnswerResult()
            answer_result.history = history
            answer_result.llm_output = {"answer": response}
            yield answer_result