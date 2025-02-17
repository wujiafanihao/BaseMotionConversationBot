import sys
import os
import torch
import torchaudio
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import threading
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Event

# 添加正确的导入路径
current_dir = Path(__file__).parent
project_root = current_dir.parent

# 添加所有必要的路径
sys.path.append(str(project_root))
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / 'CosyVoice'))
sys.path.append(str(current_dir / 'CosyVoice/third_party/Matcha-TTS'))

try:
    from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
    from cosyvoice.utils.file_utils import load_wav, logging
except ImportError as e:
    print(f"导入错误: {e}")
    print("Python 路径:", sys.path)
    print("\n当前目录结构:")
    for path in current_dir.rglob("*"):
        if path.is_file():
            print(f"  {path.relative_to(current_dir)}")
    sys.exit(1)

class AudioPlayer:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.audio_queue = Queue(maxsize=20)  # 音频队列
        self.buffer = []  # 临时缓冲区，用于存储音频片段
        self.is_playing = False
        self.play_thread = None
        self.generation_finished = Event()
        self.playback_finished = Event()  # 新增：用于标记播放完成
        
    def start_playing(self):
        """开始播放线程"""
        self.is_playing = True
        self.generation_finished.clear()
        self.playback_finished.clear()
        self.buffer = []
        self.play_thread = Thread(target=self._play_audio_thread)
        self.play_thread.daemon = True
        self.play_thread.start()
        
    def stop_playing(self):
        """停止播放"""
        self.is_playing = False
        self.generation_finished.set()
        if self.play_thread is not None:
            self.play_thread.join(timeout=1.0)
        self.buffer = []
        
    def add_audio_chunk(self, audio_chunk):
        """添加音频片段到缓冲区"""
        if not self.is_playing:
            return
            
        # 将音频数据转换为numpy数组
        if isinstance(audio_chunk, torch.Tensor):
            audio_data = audio_chunk.numpy()
        else:
            audio_data = audio_chunk
            
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            
        # 直接将音频片段添加到队列
        self.audio_queue.put(audio_data)
            
    def mark_generation_finished(self):
        """标记音频生成已完成"""
        self.generation_finished.set()
        
    def wait_for_playback(self):
        """等待所有音频播放完成"""
        self.playback_finished.wait()
        
    def _play_audio_thread(self):
        """音频播放线程"""
        try:
            while self.is_playing:
                try:
                    # 尝试获取音频片段
                    audio_data = self.audio_queue.get(timeout=0.1)
                    
                    # 播放音频
                    sd.play(audio_data, self.sample_rate)
                    sd.wait()  # 等待播放完成
                    
                except Empty:
                    # 如果队列为空且生成已完成，结束播放
                    if self.generation_finished.is_set() and self.audio_queue.empty():
                        break
                    continue
                    
                except Exception as e:
                    print(f"播放音频时出错: {str(e)}")
                    
        except Exception as e:
            print(f"音频播放线程出错: {str(e)}")
        finally:
            self.playback_finished.set()  # 标记播放完成

class TTSSystem:
    # 定义可用的模型配置
    AVAILABLE_MODELS = {
        "cosyvoice2-0.5b": {
            "path": "CosyVoice/pretrained_models/CosyVoice2-0.5B",
            "type": "cosyvoice2",
            "description": "CosyVoice2 0.5B 通用模型"
        },
        "cosyvoice-300m": {
            "path": "CosyVoice/pretrained_models/CosyVoice-300M",
            "type": "cosyvoice",
            "description": "CosyVoice 300M 通用模型"
        },
        "cosyvoice-300m-25hz": {
            "path": "pretrained_models/CosyVoice-300M-25Hz",
            "type": "cosyvoice",
            "description": "CosyVoice 300M 25Hz 采样率模型"
        },
        "cosyvoice-300m-sft": {
            "path": "CosyVoice/pretrained_models/CosyVoice-300M-SFT",
            "type": "cosyvoice",
            "description": "CosyVoice 300M SFT 微调模型"
        },
        "cosyvoice-300m-instruct": {
            "path": "pretrained_models/CosyVoice-300M-Instruct",
            "type": "cosyvoice",
            "description": "CosyVoice 300M 指令模型"
        },
        "cosyvoice-ttsfrd": {
            "path": "CosyVoice/pretrained_models/CosyVoice-ttsfrd",
            "type": "cosyvoice",
            "description": "CosyVoice TTS-FRD 模型"
        }
    }

    def __init__(self, model_name="cosyvoice2-0.5b", use_fp16=False):
        """初始化TTS系统
        Args:
            model_name: 模型名称，必须是 AVAILABLE_MODELS 中的一个
            use_fp16: 是否使用半精度
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"不支持的模型名称: {model_name}，可用模型: {list(self.AVAILABLE_MODELS.keys())}")
        
        model_config = self.AVAILABLE_MODELS[model_name]
        model_path = str(current_dir / model_config["path"])
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        self.model_type = model_config["type"]
        if self.model_type == "cosyvoice2":
            self.model = CosyVoice2(model_path, load_jit=False, load_trt=False, fp16=use_fp16)
        else:
            self.model = CosyVoice(model_path, load_jit=False, load_trt=False, fp16=use_fp16)
            
        self.sample_rate = self.model.sample_rate
        self.player = AudioPlayer(sample_rate=self.sample_rate)
        self.current_model = model_name

    @classmethod
    def get_available_models(cls):
        """获取所有可用的模型列表
        Returns:
            list: 包含模型信息的字典列表
        """
        return [
            {
                "name": name,
                "path": config["path"],
                "type": config["type"],
                "description": config["description"]
            }
            for name, config in cls.AVAILABLE_MODELS.items()
        ]

    def get_current_model(self):
        """获取当前使用的模型信息
        Returns:
            dict: 当前模型的配置信息
        """
        return {
            "name": self.current_model,
            **self.AVAILABLE_MODELS[self.current_model]
        }

    def list_speakers(self):
        """列出所有可用的说话人"""
        return self.model.list_available_spks()
        
    def stream_save_audio(self, generator, save_path):
        """生成并保存音频，确保所有片段生成完成后再播放
        Args:
            generator: 音频生成器
            save_path: 保存路径
        Returns:
            bool: 是否成功
        """
        try:
            # 存储所有音频片段
            audio_chunks = []
            
            # 生成所有音频片段
            print("正在生成音频...")
            for i, chunk in enumerate(generator):
                audio_chunk = chunk['tts_speech']
                print(f"\r已生成 {i+1} 个音频片段", end="")
                audio_chunks.append(audio_chunk)
            print("\n音频生成完成！")
            
            if audio_chunks:
                # 合并所有音频片段
                full_audio = torch.cat(audio_chunks, dim=1)
                
                # 保存完整音频
                torchaudio.save(save_path, full_audio, self.sample_rate)
                print(f"音频已保存至: {save_path}")
                
                # 播放完整音频
                audio_data = full_audio.squeeze(0).numpy()
                sd.play(audio_data, self.sample_rate)
                sd.wait()  # 等待播放完成
                
                return True
            else:
                print("未生成任何音频")
                return False
                
        except Exception as e:
            print(f"音频生成或播放出错: {str(e)}")
            return False
        
    def zero_shot_tts(self, text, prompt_wav_path, prompt_text, output_path, speed=1.0):
        """零样本语音合成"""
        if not Path(prompt_wav_path).exists():
            raise FileNotFoundError(f"参考音频文件不存在: {prompt_wav_path}")
            
        prompt_speech = load_wav(str(prompt_wav_path), 16000)
        generator = self.model.inference_zero_shot(
            tts_text=text,
            prompt_text=prompt_text,
            prompt_speech_16k=prompt_speech,
            stream=True,
            speed=speed
        )
        return self.stream_save_audio(generator, output_path)
        
    def cross_lingual_synthesis(self, text, prompt_wav_path, output_path, stream=True, speed=1.0):
        """跨语言语音合成（适用于两种模型）
        Args:
            text: 要合成的文本
            prompt_wav_path: 参考音频路径
            output_path: 输出路径
            stream: 是否使用流式生成
            speed: 语速
        """
        try:
            prompt_speech = load_wav(str(prompt_wav_path), 16000)
            
            # 根据模型类型选择不同的处理方式
            if self.model_type == "cosyvoice":
                # CosyVoice 模型的处理方式
                generator = self.model.inference_cross_lingual(
                    text,
                    prompt_speech,
                    stream=stream,
                    speed=speed
                )
            else:
                # CosyVoice2 模型的处理方式
                generator = self.model.inference_cross_lingual(
                    tts_text=text,
                    prompt_speech_16k=prompt_speech,
                    stream=stream,
                    speed=speed
                )
            
            if stream:
                return self.stream_save_audio(generator, output_path)
            else:
                # 非流式处理
                audio_chunks = []
                for chunk in generator:
                    audio_chunks.append(chunk['tts_speech'])
                if audio_chunks:
                    full_audio = torch.cat(audio_chunks, dim=1)
                    torchaudio.save(output_path, full_audio, self.sample_rate)
                    return True
                return False
                
        except Exception as e:
            print(f"跨语言合成出错: {str(e)}")
            return False

    def instruct_synthesis(self, text, character_prompt, instruct_text, prompt_wav_path=None, output_path=None, stream=True, speed=1.0):
        """指令控制语音合成（仅适用于 CosyVoice）
        Args:
            text: 要合成的文本
            character_prompt: 角色提示词（如"中文男"）
            instruct_text: 角色描述文本
            prompt_wav_path: 参考音频路径（可选）
            output_path: 输出路径（可选）
            stream: 是否使用流式生成
            speed: 语速
        """
        if self.model_type != "cosyvoice":
            raise ValueError("指令控制合成仅支持 CosyVoice 模型")
            
        try:
            # 准备参数
            kwargs = {
                "text": text,
                "character_prompt": character_prompt,
                "instruct_text": instruct_text,
                "stream": stream,
                "speed": speed
            }
            
            # 如果提供了参考音频，则加载
            if prompt_wav_path:
                kwargs["prompt_speech_16k"] = load_wav(str(prompt_wav_path), 16000)
            
            # 生成音频
            generator = self.model.inference_instruct(**kwargs)
            
            if output_path:
                if stream:
                    return self.stream_save_audio(generator, output_path)
                else:
                    # 非流式处理
                    audio_chunks = []
                    for chunk in generator:
                        audio_chunks.append(chunk['tts_speech'])
                    if audio_chunks:
                        full_audio = torch.cat(audio_chunks, dim=1)
                        torchaudio.save(output_path, full_audio, self.sample_rate)
                        return True
            return False
                
        except Exception as e:
            print(f"指令控制合成出错: {str(e)}")
            return False

    def synthesize_and_play(self, text, prompt_wav_path, output_path):
        """合成完整音频并播放
        Args:
            text: 要合成的文本
            prompt_wav_path: 参考音频路径
            output_path: 输出音频路径
        Returns:
            bool: 是否成功
        """
        try:
            # 确保输出目录存在
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 加载参考音频
            prompt_speech = load_wav(str(prompt_wav_path), 16000)
            
            # 预处理文本，减少分块
            text = text.strip()
            sentences = text.split('。')
            processed_text = []
            current_sentence = ""
            
            # 合并短句子
            for sentence in sentences:
                if not sentence.strip():
                    continue
                if len(current_sentence) + len(sentence) < 50:
                    current_sentence += sentence + "。"
                else:
                    if current_sentence:
                        processed_text.append(current_sentence)
                    current_sentence = sentence + "。"
            if current_sentence:
                processed_text.append(current_sentence)
            
            # 存储所有音频片段
            all_audio_chunks = []
            
            # 生成每个文本块的音频
            for text_block in processed_text:
                generator = self.model.inference_cross_lingual(
                    tts_text=text_block,
                    prompt_speech_16k=prompt_speech,
                    stream=True,
                    speed=1.0
                )
                
                # 收集这个文本块的所有音频片段
                block_chunks = []
                for chunk in generator:
                    audio_chunk = chunk['tts_speech']
                    block_chunks.append(audio_chunk)
                
                # 合并这个文本块的音频片段
                if block_chunks:
                    block_audio = torch.cat(block_chunks, dim=1)
                    all_audio_chunks.append(block_audio)
            
            # 合并所有文本块的音频
            if all_audio_chunks:
                full_audio = torch.cat(all_audio_chunks, dim=1)
                # 保存完整音频
                torchaudio.save(str(output_path), full_audio, self.sample_rate)
                print(f"\n音频已保存至: {str(output_path)}")
                
                # 播放完整音频
                audio_data = full_audio.squeeze(0).numpy()
                sd.play(audio_data, self.sample_rate)
                sd.wait()  # 等待播放完成
                
                return True
            else:
                print("\n未生成任何音频")
                return False
                
        except Exception as e:
            print(f"音频合成或播放出错: {str(e)}")
            return False

# def main():
#     try:
#         # 初始化TTS系统
#         tts = TTSSystem()
#         print(f"可用的说话人列表: {tts.list_speakers()}")
        
#         # 创建输出目录
#         output_dir = current_dir / "output_audio"
#         os.makedirs(output_dir, exist_ok=True)
        
#         # 参考音频路径
#         prompt_wav_path = current_dir / 'CosyVoice/asset/zero_shot_prompt.wav'
        
#         # 2. 跨语言合成示例
#         print("\n2. 跨语言合成示例")
#         tts.cross_lingual_tts(
#             text='在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]，[sad]我想你了宝贝',
#             prompt_wav_path=str(prompt_wav_path),
#             output_path=str(output_dir / "cross_lingual_stream.wav"),
#             speed=1.0
#         )
        
#         print("\n所有音频生成完成！")
        
#     except Exception as e:
#         print(f"\n错误: {str(e)}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()