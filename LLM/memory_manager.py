import chromadb
from typing import List, Dict, Any
import numpy as np
import uuid
import yaml
from abc import ABC, abstractmethod
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_ollama.embeddings import OllamaEmbeddings
import os

# 配置日志打印
def debug_print(title, content):
    print(f"\n=== {title} ===\n{content}\n{'='*(len(title)+4)}\n")

class MemoryManager:
    def __init__(self, embedding_model):
        self.client = chromadb.Client()
        try:
            # 确保存储目录存在
            os.makedirs("./memory", exist_ok=True)
            
            # 创建持久化客户端
            self.client = chromadb.PersistentClient(path="./memory")
            debug_print("向量数据库初始化", f"使用存储路径: {os.path.abspath('./memory')}")

            # 改进的集合加载逻辑
            try:
                self.collection = self.client.get_collection("chat_history")
                debug_print("集合加载", "成功加载现有对话历史集合")
            except chromadb.exceptions.NotFoundError:
                self.collection = self.client.create_collection("chat_history")
                debug_print("集合创建", "新建对话历史集合")
            except Exception as e:
                raise RuntimeError(f"集合操作异常: {str(e)}")

        except Exception as e:
            raise RuntimeError(f"无法初始化数据库: {str(e)}")
        
        self.embedding_model = embedding_model
        debug_print("嵌入模型初始化", f"使用模型: {self.embedding_model.model}")

    def _generate_embedding(self, text: str) -> List[float]:
        """生成文本嵌入向量并添加调试信息"""
        try:
            debug_print("嵌入生成", f"输入文本: {text[:50]}...")
            embedding = self.embedding_model.embed_query(text)
            debug_print("嵌入结果", f"维度: {len(embedding)} 示例: {embedding[:3]}...")
            return embedding
        except Exception as e:
            print(f"嵌入生成失败: {str(e)}")
            raise

    def store_memory(self, role: str, content: str, metadata: dict = None):
        """存储对话记录到向量数据库"""
        try:
            metadata = metadata or {}
            metadata.update({
                "id": str(uuid.uuid4()),  # 新增唯一ID
                "timestamp": datetime.now().isoformat(),
                "role": role
            })
            
            debug_print("记忆存储", f"角色: {role}\n内容: {content[:50]}...")
            embedding = self._generate_embedding(content)
            
            self.collection.add(
                ids=str(uuid.uuid4()),
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata]
            )
            debug_print("存储成功", "记录已存入向量数据库")
        except Exception as e:
            print(f"存储失败: {str(e)}")
            raise

    def retrieve_memories(self, query: str, n_results: int = 5) -> List[Dict]:
        """检索相关记忆并添加调试信息"""
        try:
            debug_print("记忆检索", f"查询: {query[:50]}...\n数量: {n_results}")
            query_embedding = self._generate_embedding(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            debug_print("原始检索结果", f"找到{len(results['documents'][0])}条记录")
            
            sorted_results = sorted([
                {
                    "content": doc,
                    "metadata": meta,
                    "similarity": dist
                } for doc, meta, dist in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ], key=lambda x: x['similarity'], reverse=True)
            
            debug_print("排序后结果", "\n".join(
                [f"[相似度 {item['similarity']:.2f}] {item['content'][:50]}..." 
                 for item in sorted_results]
            ))
            return sorted_results
        except Exception as e:
            print(f"检索失败: {str(e)}")
            raise

class EnhancedMemoryManager(MemoryManager):
    def contextual_retrieval(self, current_query: str, n_results: int = 5):
        """增强型上下文检索策略"""
        try:
            # 获取语义相关结果
            semantic_results = self.retrieve_memories(current_query, n_results)
            
            # 获取最近5条记录
            recent_memories = self.get_recent_memories(n=5)
            
            # 合并结果并去重
            combined = self._merge_results(semantic_results, recent_memories)
            
            debug_print("最终上下文", 
                f"包含{len(combined)}条记录（语义相关{len(semantic_results)} + 最近{len(recent_memories)}）")
            
            return combined[:n_results+5]  # 返回扩充后的结果

        except Exception as e:
            print(f"上下文检索失败: {str(e)}")
            raise

    def _merge_results(self, semantic_results, recent_memories):
        """合并语义结果和最近记录"""
        # 使用ID去重
        seen = set()
        merged = []
        
        # 优先保留语义相关结果
        for item in semantic_results:
            uid = item["metadata"].get("id", hash(item["content"]))
            if uid not in seen:
                seen.add(uid)
                merged.append(item)
        
        # 补充最近记录
        for item in recent_memories:
            uid = item["metadata"].get("id", hash(item["content"]))
            if uid not in seen:
                seen.add(uid)
                merged.append({
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "similarity": 0.0  # 标记为新增记录
                })
        
        # 按时间重新排序
        return sorted(
            merged,
            key=lambda x: x["metadata"].get("timestamp", ""),
            reverse=True
        )
    def get_recent_memories(self, n: int):
            """获取最近n条对话记录（带错误处理）"""
            try:
                all_memories = self.collection.get()
                if not all_memories.get('documents'):
                    debug_print("记忆检索", "数据库为空")
                    return []

                # 确保元数据包含时间戳
                valid_memories = []
                for doc, meta in zip(all_memories['documents'], all_memories['metadatas']):
                    if 'timestamp' in meta:
                        valid_memories.append((doc, meta))
                    else:
                        debug_print("数据警告", f"发现无效记录: {doc[:50]}...")

                # 按时间排序
                sorted_memories = sorted(
                    valid_memories,
                    key=lambda x: x[1]['timestamp'],
                    reverse=True
                )
                
                debug_print("最近记忆", f"找到{len(sorted_memories)}条有效记录")
                return [
                    {"content": doc, "metadata": meta} 
                    for doc, meta in sorted_memories[:n]
                ]
                
            except Exception as e:
                print(f"获取最近记忆失败: {str(e)}")
                raise

class LLMInterface(ABC):
    def __init__(self):
        self.memory = None  # 由子类初始化

    @abstractmethod
    def model(self) -> Any:
        pass

    @abstractmethod
    def generate(self, message: str, emotion: str) -> str:
        pass

class OpenAIAdapter(LLMInterface):
    _template = """
    【对话上下文（最近优先）】:
    {chat_history}
    【用户信息】:
    情绪: {emotion}
    提问: {query}
    【辅助思考】:
    {thought_process}
    【最终任务】:
    请基于以上信息生成一段简洁且富有同理心的回答。请注意回答时：
    1. 不需直接引用辅助思考内容；
    2. 如果用户情绪低落，请给予适当安慰与鼓励；
    3. 保持回复内容精炼，不废话。
    """

    def __init__(self, base_url: str = None, api_key: str = None):
        super().__init__()
        config = self.load_config()
        self.api_key = api_key or config.get("api_key")
        self.base_url = base_url or config.get("base_url")
        self.model_name = config.get("model", "gpt-4o-mini")
        
        try:
            # 初始化嵌入模型和记忆系统
            self.embedder = OllamaEmbeddings(model="bge-large:latest")
            debug_print("Ollama初始化", "BGE嵌入模型加载成功")
            self.memory = EnhancedMemoryManager(self.embedder)
            self.model_instance = self.model()
        except Exception as e:
            raise RuntimeError(f"初始化失败: {str(e)}")
        
    def __del__(self):
        try:
            if hasattr(self, 'client'):
                self.client.close()
                debug_print("资源清理", "已关闭数据库连接")
        except Exception as e:
            print(f"清理资源时出错: {str(e)}")

    @staticmethod
    def load_config(config_file="config.yaml"):
        try:
            with open(config_file, 'r', encoding="utf-8") as f:
                return yaml.safe_load(f).get("openai", {})
        except Exception as e:
            raise ValueError(f"配置文件错误: {str(e)}")

    def model(self) -> Any:
        try:
            return ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model_name
            )
        except Exception as e:
            raise RuntimeError(f"模型初始化失败: {str(e)}")

    def generate(self, message: str, emotion: str) -> str:
        try:
            # 存储用户消息
            self.memory.store_memory("user", message, {"emotion": emotion})
            
            # 检索相关记忆
            related_memories = self.memory.contextual_retrieval(message)
            chat_history = "\n".join(
                [f"{m['metadata']['role']}: {m['content']}" 
                 for m in related_memories]
            )
            
            # 生成思考过程
            thought = self._generate_thought_process(message, emotion)
            
            # 构建提示词
            prompt = self._template.format(
                chat_history=chat_history,
                emotion=emotion,
                query=message,
                thought_process=thought
            )
            debug_print("完整提示词", prompt)
            
            # 调用模型
            response = self.model_instance.invoke(prompt).content
            
            # 存储AI回复
            self.memory.store_memory("assistant", response)
            return response
        except Exception as e:
            print(f"生成失败: {str(e)}")
            return "抱歉，我遇到了一些问题，请稍后再试。"

    def _generate_thought_process(self, message: str, emotion: str) -> str:
        """生成思考过程（示例简化版）"""
        return f"用户当前情绪为 {emotion}，需要重点关注。问题核心是：{message[:50]}..."

if __name__ == "__main__":
    try:
        debug_print("存储路径", f"对话记忆将保存在: {os.path.abspath('./memory')}")
        # 初始化系统
        debug_print("系统启动", "正在初始化AI助手...")
        llm = OpenAIAdapter()
        
        # 对话循环
        debug_print("系统就绪", "可以开始对话 (输入'exit'退出)")
        while True:
            try:
                user_input = input("\n你：")
                if user_input.lower() == 'exit':
                    break
                
                emotion = input("当前情绪（默认neutral）：") or "neutral"
                response = llm.generate(user_input, emotion)
                print("\nAI：", response)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"对话出错: {str(e)}")
                
    except Exception as e:
        print(f"系统启动失败: {str(e)}")