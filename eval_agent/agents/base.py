# import logging
# from typing import List, Dict, Any, Mapping, Union

# logger = logging.getLogger("agent_frame")

# class LMAgent:
#     def __init__(self, config: Mapping[str, Any]):
#         self.config = config
#         logger.debug(f"Initialized {self.__class__.__name__} with config: {config}")
#         self.stop_words = ["\nObservation:", "\nTask:", "\n---"]

#     def __call__(self, history: List[Union[str, Dict[str, str]]]) -> str:
#         logger.debug(f"LMAgent __call__ received history: {history}")
#         try:
#             messages = self.prepare_messages(history)
#             logger.debug(f"Prepared messages: {messages}")
#             response = self.generate_response(messages)
#             logger.debug(f"Generated response: {response}")
#             return response
#         except Exception as e:
#             logger.error(f"Error in LMAgent.__call__: {e}", exc_info=True)
#             raise

#     def prepare_messages(self, history: List[Union[str, Dict[str, str]]]) -> List[Dict[str, str]]:
#         messages = []
#         for item in history:
#             if isinstance(item, str):
#                 messages.append({"role": "user", "content": item})
#             elif isinstance(item, dict) and "role" in item and "content" in item:
#                 messages.append(item)
#             else:
#                 logger.warning(f"Unexpected item in history: {item}")
#                 messages.append({"role": "user", "content": str(item)})
#         return self.add_system_message(messages)

#     def add_system_message(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
#         if not messages:
#             return [{"role": "system", "content": "You are an AI assistant. Complete the given task."}]
        
#         first_msg = messages[0]
#         if first_msg["role"] != "user":
#             return [{"role": "system", "content": "You are an AI assistant. Complete the given task."}] + messages
        
#         content = first_msg["content"]
#         if "\n---\n" in content:
#             parts = content.split("\n---\n")
#             system = parts[0]
#             task = parts[-1]
#             examples = "\n---\n".join(parts[1:-1]) if len(parts) > 2 else ""
#             return [
#                 {"role": "system", "content": system},
#                 {"role": "user", "content": (examples + "\n---\n" + task).strip()},
#             ] + messages[1:]
#         else:
#             return [
#                 {"role": "system", "content": "You are an AI assistant. Complete the given task."},
#                 {"role": "user", "content": content},
#             ] + messages[1:]

#     def generate_response(self, messages: List[Dict[str, str]]) -> str:
#         # 这个方法应该被子类重写以实现特定的模型逻辑
#         raise NotImplementedError("Subclasses must implement generate_response method")



# 这个下面肯定可以跑  但我想试一下提示词工程
import logging
from typing import List, Dict, Any, Mapping

logger = logging.getLogger("agent_frame")


class LMAgent:
    """Base class for an agent."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        logger.debug(f"Initialized {self.__class__.__name__} with config: {config}")
        # The agent should not generate observations or expert feedback
        self.stop_words = ["\nObservation:", "\nTask:", "\n---"]

    def __call__(self) -> str:
        pass

    def add_system_message(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        # Prepend the prompt with the system message
        first_msg = messages[0]
        assert first_msg["role"] == "user"
        system, examples, task = first_msg["content"].split("\n---\n")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": examples + "\n---\n" + task},
        ] + messages[1:]
        return messages
