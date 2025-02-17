from abc import ABC, abstractmethod
from typing import Any
import yaml
from langchain_openai import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from .memory_manager import debug_print, EnhancedMemoryManager
import asyncio

# 定义 LLMInterface 接口
class LLMInterface(ABC):
    def __init__(self):
        self.memory = []  # 用于存储对话记忆

    @abstractmethod
    def model(self) -> Any:
        pass

    @abstractmethod
    def generate(self, message: str, emotion: str) -> str:
        pass

# OpenAIAdapter 实现 LLMInterface
class ModelAdapter(LLMInterface):
    _template = """
    【对话上下文（最近优先）】:
    {chat_history}
    （请注意：历史记录中，assistant代表你之前给出的回答，而user代表用户提出的问题。）
    
    【用户信息】:
    情绪: {emotion}
    提问: {query}
    
    【辅助思考】:
    ----- 链式思考开始 -----
    {thought_process}
    ----- 链式思考结束 -----
    
    【最终任务】:
    请基于上述历史记录、用户当前输入和辅助思考的结果，生成一段简洁且富有同理心的回答。请注意：
    1. 不直接引用辅助思考内容；
    2. 根据语境和情感自然，使对话更生动；
    3. 如果用户情绪低落，请给予适当的安慰与鼓励；
    4. 回答内容应充分考虑历史对话中的角色信息，紧密回应用户当前的问题；
    5. 保持回复内容精炼，不废话；
    6. 回复的表达应温暖、细腻，带有充足的情感润色，体现关怀与同理心，避免生硬机械；

    """

    def __init__(self, base_url: str = None, api_key: str = None):
        super().__init__()
        # 使用 load_config 动态根据 config.yaml 中的 selected_adapter 读取对应配置
        config = self.load_config()  
        self.api_key = api_key or config.get("api_key")
        self.base_url = base_url or config.get("base_url")
        self.model_name = config.get("model", "gemini-2.0-pro-exp-02-05")
        self.provider = config.get("selected_adapter", "gemini")  # 添加 provider 属性
        debug_print("配置信息", f"Provider: {self.provider}, Model: {self.model_name}, API Key: {self.api_key[:8]}...")
        try:
            # 初始化模型实例
            self.model_instance = self.model()
            debug_print("模型初始化", "模型实例创建成功")
            
            # 初始化嵌入模型和记忆系统
            self.embedder = OllamaEmbeddings(
                model="bge-large:latest",
            )
            debug_print("Ollama初始化", "BGE嵌入模型加载成功")
            self.memory = EnhancedMemoryManager(self.embedder)
            
            # 初始化 ReasonerAdapter
            self.reasoner = ReasonerAdapter()
            debug_print("初始化完成", "所有组件加载成功")
        except Exception as e:
            debug_print("初始化错误", f"错误详情: {str(e)}")
            raise RuntimeError(f"初始化失败: {str(e)}")
    def model(self) -> Any:
        try:
            if self.provider.lower() == "openai":
                return ChatOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    model=self.model_name
                )
            elif self.provider.lower() == "gemini":
                return ChatOpenAI(
                    model=self.model_name,
                    base_url=self.base_url,
                    api_key=self.api_key,
                    temperature=0.7
                )
            else:
                raise ValueError(f"不支持的模型提供商: {self.provider}")
        except Exception as e:
            debug_print("模型初始化错误", f"错误详情: {str(e)}")
            raise RuntimeError(f"模型初始化失败: {str(e)}")
    def update_config(self, provider: str, model_name: str, thinking_model: str) -> None:
        """更新模型配置并重新初始化模型
        Args:
            provider: 模型提供商 ("OpenAI" 或 "Gemini")
            model_name: 选择的模型名称
            thinking_model: 思考模型名称
        """
        try:
            # 更新配置文件
            with open("config.yaml", 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 更新选定的适配器
            config['selected_adapter'] = provider.lower()
            
            # 更新对应提供商的配置
            provider_config = config.get(provider.lower(), {})
            provider_config['model'] = model_name
            config[provider.lower()] = provider_config
            
            # 更新思考模型配置
            deepseek_config = config.get('deepseek', {})
            deepseek_config['model'] = thinking_model
            config['deepseek'] = deepseek_config
            
            # 保存更新后的配置
            with open("config.yaml", 'w', encoding='utf-8') as f:
                yaml.safe_dump(config, f, allow_unicode=True)
            
            # 更新当前实例的配置
            self.provider = provider.lower()
            self.model_name = model_name
            
            # 重新初始化模型实例
            self.model_instance = self.model()
            
            # 重新初始化思考模型
            if hasattr(self, 'reasoner'):
                self.reasoner = ReasonerAdapter()
                
            debug_print("配置更新", f"已更新为 {provider} 的 {model_name} 模型")
                
        except Exception as e:
            raise RuntimeError(f"更新配置失败: {str(e)}")

    @classmethod
    def load_config(cls, config_file="config.yaml"):
        """加载聊天模型配置
        Returns:
            dict: 当前选定适配器的配置
        """
        try:
            with open(config_file, 'r', encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            # 获取当前选定的适配器
            selected_adapter = config.get("selected_adapter", "gemini")
            
            # 获取对应适配器的配置
            if selected_adapter == "gemini":
                adapter_config = config.get("gemini", {})
            elif selected_adapter == "openai":
                adapter_config = config.get("openai", {})
            else:
                raise ValueError(f"不支持的适配器类型: {selected_adapter}")
            
            # 添加 selected_adapter 到配置中
            adapter_config["selected_adapter"] = selected_adapter
            
            debug_print("配置加载", f"已加载 {selected_adapter} 配置")
            return adapter_config
            
        except Exception as e:
            raise ValueError(f"Failed to load config file: {e}")

    def generate(self, message: str, emotion: str) -> str:
        try:
            # 获取已有 ReasonerAdapter 的思考过程和历史记录
            think, chat_memories = self.reasoner.generate(message, emotion)
            
            # 直接使用 ReasonerAdapter 检索到的历史记录
            chat_history = "\n".join(
                [f"{m['metadata']['role']}: {m['content']}" 
                for m in chat_memories]
            )

            # 使用 format 方法生成最终的提示
            prompt = self._template.format(
                chat_history=chat_history,
                emotion=emotion,
                query=message,
                thought_process=think
            )

            debug_print("完整提示词", prompt)

            # 调用实际的模型
            try:
                response = self.model_instance.invoke(prompt).content
            except Exception as e:
                raise RuntimeError(f"Model invocation failed: {e}")

            # 在生成回复后，统一存储用户输入和AI回复
            self.memory.store_memory("user", message, {"emotion": emotion})
            self.memory.store_memory("assistant", response)
            
            return response, think
        except Exception as e:
            print(f"生成失败: {str(e)}")
            return "抱歉，我遇到了一些问题，请稍后再试。", "思考过程出错"

    async def agenerate(self, message: str, emotion: str) -> str:
        """异步生成回复的方法，支持流式传输并实时打印输出，最终返回完整响应。"""
        # 构建包含记忆的上下文
        chat_history = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in self.memory]
        )

        # 获取已有 ReasonerAdapter 的思考过程（避免重复初始化，加快响应速度）
        think = self.reasoner.generate(message, emotion)

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
        # 在生成回复后，统一添加用户消息及AI回复到记忆中
        self.memory.append({"role": "user", "content": message})
        self.memory.append({"role": "assistant", "content": final_response})
        return final_response


# ReasonerAdapter 实现 LLMInterface（Ollama 适配器）
class ReasonerAdapter(LLMInterface):
    _template = """
    【历史对话记录】:
    {chat_history}
    （请注意：在历史记录中，assistant代表你之前的发言，而user代表用户的输入。）
    
    【当前输入】:
    情绪: {emotion}
    提问: {query}
    
    【生成任务】:
    请基于以上历史记录和当前输入，生成一段详细的链式思考。该链式思考应：
    1. 分析对话历史中的信息，明确区分assistant和user的发言；
    2. 提炼出用户的核心需求、情绪变化及可能的目的；
    3. 为后续生成符合用户情绪和需求的回复提供指导思路；

    【思考格式】:
    示例思考格式：
    1. 分析用户情绪：用户当前表现出{emotion}的情绪
    2. 历史对话分析：...
    3. 核心需求提炼：...
    """

    def __init__(self):
        super().__init__()
        config = self.load_config()
        self.model_name = config.get("model", "deepseek-r1-1.5b")
        self.base_url = config.get("base_url", "http://127.0.0.1:11434")
        self.model_instance = self.model()  # 初始化模型实例
        try:
            # 初始化向量数据库记忆系统
            self.embedder = OllamaEmbeddings(model="bge-large:latest")
            self.memory = EnhancedMemoryManager(self.embedder)
        except Exception as e:
            debug_print("ReasonerAdapter内存初始化", f"初始化失败: {str(e)}")
            self.memory = None

    @classmethod
    def load_config(cls, config_file="config.yaml"):
        """加载思考模型配置
        Returns:
            dict: deepseek 配置
        """
        try:
            with open(config_file, 'r', encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            # 获取 deepseek 配置
            deepseek_config = config.get("deepseek", {})
            
            # 设置默认值
            if not deepseek_config:
                deepseek_config = {
                    "base_url": "http://127.0.0.1:11434",
                    "model": "deepseek-r1:8b"
                }
            
            return deepseek_config
        except Exception as e:
            raise ValueError(f"Failed to load config file: {e}")
    def model(self) -> Any:
        return ChatOllama(model=self.model_name, temperature=0.7, base_url=self.base_url)

    def generate(self, message: str, emotion: str) -> tuple:
        """生成思考过程和检索历史
        Args:
            message: 用户消息
            emotion: 情绪标签
        Returns:
            tuple: (思考过程, 聊天历史列表)
        """
        # 检索向量数据库中的历史对话记录
        chat_memories = []
        if hasattr(self, "memory") and self.memory is not None:
            chat_memories = self.memory.contextual_retrieval(message)
        chat_history = "\n".join([f"{m['metadata']['role']}: {m['content']}" for m in chat_memories]) if chat_memories else "无历史记录"

        # 使用 format 方法生成最终的提示，并将历史对话记录包含在内
        prompt = self._template.format(chat_history=chat_history, emotion=emotion, query=message)

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
            
            # 返回思考内容和检索到的历史记录
            return think_content, chat_memories
        except Exception as e:
            raise RuntimeError(f"Model invocation failed: {e}")