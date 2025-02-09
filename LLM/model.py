from abc import ABC, abstractmethod
from typing import Any
import yaml
from langchain_openai import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama
import asyncio

# 定义 LLMInterface 接口
class LLMInterface(ABC):
    def __init__(self):
        self.memory = []  # 用于存储对话记忆

    @abstractmethod
    def model(self) -> Any:
        """
        根据用户配置初始化模型
        :return: 初始化后的模型
        """
        pass

    @abstractmethod
    def generate(self, message: str, emotion: str) -> str:
        """
        根据输入的提示生成文本，同时利用记忆生成上下文。
        :param message: 用户输入的提示字符串
        :param emotion: 用户当前的情绪
        :return: 生成的文本字符串
        """
        pass

# OpenAIAdapter 实现 LLMInterface
class OpenAIAdapter(LLMInterface):
    _template = """
    【对话上下文】:
    {chat_history}
    【用户信息】:
    情绪: {emotion}
    提问: {query}
    【辅助思考】:
    ----- 链式思考开始 -----
    {thought_process}
    ----- 链式思考结束 -----
    【最终任务】:
    请基于以上信息生成一段简洁且富有同理心的回答。请注意回答时：
    1. 不需直接引用辅助思考内容；
    2. 如果用户情绪低落，请给予适当安慰与鼓励；
    3. 保持回复内容精炼，不废话。
    """

    def __init__(self, base_url: str = None, api_key: str = None):
        super().__init__()
        config = self.load_config()  # 从yaml配置读取
        self.api_key = api_key or config.get("api_key")
        self.base_url = base_url or config.get("base_url")
        self.model_name = config.get("model", "gpt-4o-mini")
        self.model_instance = self.model()  # 初始化模型实例

    @staticmethod
    def load_config(config_file="config.yaml"):
        try:
            with open(config_file, 'r', encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config.get("gemini", {})
        except Exception as e:
            raise ValueError(f"Failed to load config file: {e}")

    def model(self) -> Any:
        return ChatOpenAI(api_key=self.api_key, base_url=self.base_url, model=self.model_name)

    def generate(self, message: str, emotion: str) -> str:
        # 将用户消息添加到记忆中
        self.memory.append({"role": "user", "content": message})

        # 构建包含记忆的上下文
        chat_history = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in self.memory]
        )

        # 获取 ReasonerAdapter 的思考过程
        think = ReasonerAdapter().generate(message, emotion)

        # 使用 format 方法生成最终的提示
        prompt = self._template.format(
            chat_history=chat_history,
            emotion=emotion,
            query=message,
            thought_process=think
        )

        # 调用实际的 OpenAI 模型
        try:
            response = self.model_instance.invoke(prompt).content
        except Exception as e:
            raise RuntimeError(f"Model invocation failed: {e}")

        # 将模型响应添加到记忆中
        self.memory.append({"role": "assistant", "content": response})
        return response

    async def agenerate(self, message: str, emotion: str) -> str:
        """异步生成回复的方法，支持流式传输并实时打印输出，最终返回完整响应。"""
        # 将用户消息添加到记忆中
        self.memory.append({"role": "user", "content": message})

        # 构建包含记忆的上下文
        chat_history = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in self.memory]
        )

        # 获取 ReasonerAdapter 的思考过程（这里同步调用，如需要也可异步化）
        think = ReasonerAdapter().generate(message, emotion)

        # 使用 format 方法生成最终的提示
        prompt = self._template.format(
            chat_history=chat_history,
            emotion=emotion,
            query=message,
            thought_process=think
        )

        accumulated = []
        try:
            # 异步调用模型接口，假设 model_instance.astream(prompt) 返回一个异步生成器
            async for token in self.model_instance.astream(prompt):
                token_str = token.content if hasattr(token, "content") else str(token)
                print(token_str, end='', flush=True)
                accumulated.append(token_str)
        except Exception as e:
            raise RuntimeError(f"Async model invocation failed: {e}")

        final_response = ''.join(accumulated)
        self.memory.append({"role": "assistant", "content": final_response})
        return final_response


# ReasonerAdapter 实现 LLMInterface（Ollama 适配器）
class ReasonerAdapter(LLMInterface):
    _template = """
    【输入信息】:
    情绪: {emotion}
    用户提问: {query}
    【任务】:
    请生成一段简洁的链式思考，用于辅助后续回答的生成。该思考应提炼出用户意图的核心要点，并提供简洁的情绪应对提示，仅供内部使用，不直接展示给用户。
    """

    def __init__(self):
        super().__init__()
        config = self.load_config()
        self.model_name = config.get("model", "deepseek-r1-1.5b")
        self.base_url = config.get("base_url", "http://127.0.0.1:11434")
        self.model_instance = self.model()  # 初始化模型实例

    def model(self) -> Any:
        return ChatOllama(model=self.model_name, temperature=0.7, base_url=self.base_url)

    @staticmethod
    def load_config(config_file="config.yaml"):
        try:
            with open(config_file, 'r', encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config.get("deepseek", {})
        except Exception as e:
            raise ValueError(f"Failed to load config file: {e}")

    def generate(self, message: str, emotion: str) -> str:
        # 使用 format 方法生成最终的提示
        prompt = self._template.format(emotion=emotion, query=message)

        try:
            # 调用模型接口，并传入 stream=True 以启用流式输出
            generator = self.model_instance.stream(prompt)
            accumulated = []
            in_think_block = False  # 标记是否在 <think> 块内
            print("<think>", end='', flush=True)
            for chunk in generator:
                if hasattr(chunk, "content"):  # 确保 chunk 包含 content 属性
                    token = chunk.content
                else:
                    token = str(chunk)  # 如果没有 content 属性，直接转换为字符串
                
                # 检测 <think> 开始
                if "<think>" in token:
                    in_think_block = True
                    token = token.replace("<think>", "")  # 移除 <think> 标签
                    print(token, end='', flush=True)

                # 检测 </think> 结束
                if "</think>" in token:
                    in_think_block = False
                    token = token.replace("</think>", "")  # 移除 </think> 标签
                    print(token, end='', flush=True)

                if in_think_block:
                    print(token, end='', flush=True)
                    accumulated.append(token)  # 只在 <think> 块内累积内容

            # 拼接最终的思考内容
            think_content = ''.join(accumulated).strip()
            print("\n</think>\n", flush=True)
        except Exception as e:
            raise RuntimeError(f"Model invocation failed: {e}")
        return think_content

async def main():
    openai_model = OpenAIAdapter()
    response = await openai_model.agenerate("你好", "高兴")
    print("AI：", response)
    response = await openai_model.agenerate("我刚刚问了什么", "高兴")
    print("AI：", response)

if __name__ == "__main__":
    try:
        openai_model = OpenAIAdapter()
        reasoner_model = ReasonerAdapter()
        # while True:
        #     # 用户提问
        #     user_message = input("你：")
        #     user_emotion = input("你当前情绪：")
        #     # 基于 ReasonerAdapter 的思考过程生成最终回复
        #     response = openai_model.generate(user_message, user_emotion)
        #     print("AI：", response)
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}")