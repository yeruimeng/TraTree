# import json
# import time
# import logging
# from typing import List, Dict, Union, Any
# import requests
# from fastchat.model.model_adapter import get_conversation_template
# from requests.exceptions import Timeout, ConnectionError
# from .base import LMAgent

# logger = logging.getLogger("agent_frame")

# def _add_to_set(s, new_stop):
#     if not s:
#         return
#     if isinstance(s, str):
#         new_stop.add(s)
#     else:
#         new_stop.update(s)

# class FastChatAgent(LMAgent):
#     def __init__(self, config) -> None:
#         super().__init__(config)
#         self.controller_address = config["controller_address"]
#         self.model_name = config["model_name"]
#         self.temperature = config.get("temperature", 0)
#         self.max_new_tokens = config.get("max_new_tokens", 512)
#         self.top_p = config.get("top_p", 0)

#     def generate_response(self, messages: List[Dict[str, str]]) -> str:
#         controller_addr = self.controller_address
#         worker_addr = controller_addr
#         if worker_addr == "":
#             raise ValueError("Worker address is empty")

#         gen_params = {
#             "model": self.model_name,
#             "temperature": self.temperature,
#             "max_new_tokens": self.max_new_tokens,
#             "echo": False,
#             "top_p": self.top_p,
#         }

#         conv = get_conversation_template(self.model_name)
#         for message in messages:
#             role = message["role"]
#             content = message["content"]
#             if role == "user":
#                 conv.append_message(conv.roles[0], content)
#             elif role == "assistant":
#                 conv.append_message(conv.roles[1], content)
#             elif role == "system":
#                 conv.set_system_message(content)

#         conv.append_message(conv.roles[1], None)
#         prompt = conv.get_prompt()

#         new_stop = set()
#         _add_to_set(self.stop_words, new_stop)
#         _add_to_set(conv.stop_str, new_stop)

#         gen_params.update({
#             "prompt": prompt,
#             "stop": list(new_stop),
#             "stop_token_ids": conv.stop_token_ids,
#         })

#         headers = {"User-Agent": "FastChat Client"}

#         for _ in range(3):
#             try:
#                 response = requests.post(
#                     controller_addr + "/worker_generate_stream",
#                     headers=headers,
#                     json=gen_params,
#                     stream=True,
#                     timeout=120,
#                 )
#                 text = ""
#                 for line in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
#                     if line:
#                         data = json.loads(line)
#                         if data["error_code"] != 0:
#                             raise Exception(data["text"])
#                         text = data["text"]
#                 return text
#             except Timeout:
#                 logger.warning("Timeout, retrying...")
#             except ConnectionError:
#                 logger.warning("Connection error, retrying...")
#             time.sleep(5)
        
#         raise Exception("Timeout after 3 retries.")

#     def prepare_messages(self, history: List[str]) -> List[Dict[str, str]]:
#         messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": msg} for i, msg in enumerate(history)]
#         return self.add_system_message(messages)

#     def add_system_message(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
#         first_msg = messages[0]
#         assert first_msg["role"] == "user"
        
#         if "\n---\n" in first_msg["content"]:
#             system, examples, task = first_msg["content"].split("\n---\n")
#             messages = [
#                 {"role": "system", "content": system},
#                 {"role": "user", "content": examples + "\n---\n" + task},
#             ] + messages[1:]
#         else:
#             messages = [
#                 {"role": "system", "content": "You are an AI assistant. Complete the given task."},
#                 {"role": "user", "content": first_msg["content"]},
#             ] + messages[1:]
        
#         return messages





import json
import time
import logging
from typing import List, Dict, Union, Any
import requests
from fastchat.model.model_adapter import get_conversation_template
from requests.exceptions import Timeout, ConnectionError

from .base import LMAgent

logger = logging.getLogger("agent_frame")


def _add_to_set(s, new_stop):
    if not s:
        return
    if isinstance(s, str):
        new_stop.add(s)
    else:
        new_stop.update(s)


class FastChatAgent(LMAgent):
    """This agent is a test agent, which does nothing. (return empty string for each action)"""

    def __init__(
        self,
        config
    ) -> None:
        super().__init__(config)
        self.controller_address = config["controller_address"]
        self.model_name = config["model_name"]
        self.temperature = config.get("temperature", 0.0)
        self.max_new_tokens = config.get("max_new_tokens", 512)
        self.top_p = config.get("top_p", 0.0)

    def __call__(self, messages: List[dict]) -> str:
        controller_addr = self.controller_address
        worker_addr = controller_addr
        if worker_addr == "":
            raise ValueError
        gen_params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "echo": False,
            "top_p": self.top_p,
        }
        conv = get_conversation_template(self.model_name)
        for history_item in messages:
            role = history_item["role"]
            content = history_item["content"]
            if role == "user":
                conv.append_message(conv.roles[0], content)
            elif role == "assistant":
                conv.append_message(conv.roles[1], content)
            else:
                raise ValueError(f"Unknown role: {role}")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        new_stop = set()
        _add_to_set(self.stop_words, new_stop)
        _add_to_set(conv.stop_str, new_stop)
        gen_params.update(
            {
                "prompt": prompt,
                "stop": list(new_stop),
                "stop_token_ids": conv.stop_token_ids,
            }
        )
        headers = {"User-Agent": "FastChat Client"}
        for _ in range(3):
            try:
                response = requests.post(
                    controller_addr + "/worker_generate_stream",
                    headers=headers,
                    json=gen_params,
                    stream=True,
                    timeout=120,
                )
                text = ""
                for line in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                    if line:
                        data = json.loads(line)
                        if data["error_code"] != 0:
                            assert False, data["text"]
                        text = data["text"]
                return text
            # if timeout or connection error, retry
            except Timeout:
                print("Timeout, retrying...")
            except ConnectionError:
                print("Connection error, retrying...")
            time.sleep(5)
        else:
            raise Exception("Timeout after 3 retries.")




