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
    def __init__(self, model_path=None, use_fp16=False):
        """初始化TTS系统"""
        if model_path is None:
            model_path = str(current_dir / 'CosyVoice/pretrained_models/CosyVoice2-0.5B')
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
            
        self.model = CosyVoice2(model_path, load_jit=False, load_trt=False, fp16=use_fp16)
        self.sample_rate = self.model.sample_rate
        self.player = AudioPlayer(sample_rate=self.sample_rate)
        
    def list_speakers(self):
        """列出所有可用的说话人"""
        return self.model.list_available_spks()
        
    def stream_save_audio(self, generator, save_path):
        """流式保存并播放音频，采用流水线模式：
        初始阶段等待生成4个音频片段播放，之后每生成2个片段播放一次。
        播放过程中生成继续进行，从而避免播放卡壳问题。
        """
        audio_chunks = []  # 保存所有音频片段用于最终合并保存
        shared_buffer = []  # 共享缓冲区，用于流水线播放（存 squeeze 后的音频片段）
        lock = threading.Lock()
        generation_done = Event()
        
        def generation_pipeline():
            try:
                for i, chunk in enumerate(generator):
                    audio_chunk = chunk['tts_speech']
                    print(f"\r正在生成第 {i+1} 个音频片段...", end="")
                    audio_chunks.append(audio_chunk)
                    # 将音频片段 squeeze 后放入共享缓冲区
                    with lock:
                        shared_buffer.append(audio_chunk.squeeze(0))
                    time.sleep(0.2)  # 生成间隔，200ms
            except Exception as e:
                print(f"\n生成音频时出错: {str(e)}")
            finally:
                print("\n生成完成，等待播放结束...")
                generation_done.set()
        
        def playback_pipeline():
            first_batch = True  # 标记是否为初始阶段
            while True:
                with lock:
                    available = len(shared_buffer)
                    # 如果生成完成且无剩余数据，则退出
                    if generation_done.is_set() and available == 0:
                        break
                    
                    if generation_done.is_set() and 0 < available < 4:
                        # 生成完成，且有剩余片段，全部播放完
                        batch = [shared_buffer.pop(0) for _ in range(available)]
                    else:
                        # 正常的批次处理逻辑
                        threshold = 4 if first_batch else 2
                        if available >= threshold:
                            batch = [shared_buffer.pop(0) for _ in range(threshold)]
                        else:
                            batch = None
                            
                if batch:
                    # 合并批次的音频片段
                    merged = np.concatenate(batch)
                    sd.play(merged, self.sample_rate)
                    sd.wait()
                    if first_batch:
                        first_batch = False
                else:
                    # 如果没有数据可播放，检查是否应该退出
                    if generation_done.is_set() and len(shared_buffer) == 0:
                        break
                    time.sleep(0.1)
        
        # 启动生成和播放线程
        gen_thread = Thread(target=generation_pipeline)
        play_thread = Thread(target=playback_pipeline)
        gen_thread.start()
        play_thread.start()
        
        # 等待生成和播放全部完成
        gen_thread.join()
        play_thread.join()
        
        if audio_chunks:
            # 保存生成的完整音频
            full_audio = torch.cat(audio_chunks, dim=1)
            torchaudio.save(save_path, full_audio, self.sample_rate)
            print(f"\n音频已保存至: {save_path}")
            return True
        else:
            print("\n未生成任何音频")
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
        
    def cross_lingual_tts(self, text, prompt_wav_path, output_path, speed=1.0):
        """跨语言语音合成"""
        if not Path(prompt_wav_path).exists():
            raise FileNotFoundError(f"参考音频文件不存在: {prompt_wav_path}")
            
        prompt_speech = load_wav(str(prompt_wav_path), 16000)
        
        # 预处理文本，减少分块
        text = text
        sentences = text.split('。')  # 按句号分割
        processed_text = []
        current_sentence = ""
        
        # 合并短句子
        for sentence in sentences:
            if not sentence.strip():
                continue
            if len(current_sentence) + len(sentence) < 50:  # 调整这个阈值以控制分块大小
                current_sentence += sentence + "。"
            else:
                if current_sentence:
                    processed_text.append(current_sentence)
                current_sentence = sentence + "。"
        if current_sentence:
            processed_text.append(current_sentence)
            
        # 生成每个文本块的音频
        for i, text_block in enumerate(processed_text):
            generator = self.model.inference_cross_lingual(
                tts_text=text_block,
                prompt_speech_16k=prompt_speech,
                stream=True,
                speed=speed
            )
            if i == 0:
                # 第一个块，创建新的输出文件
                self.stream_save_audio(generator, output_path)
            else:
                # 后续块，追加到现有文件
                temp_path = output_path.parent / f"temp_{i}.wav"
                self.stream_save_audio(generator, temp_path)
                # 合并音频文件
                # TODO: 实现音频文件合并逻辑
                
        return True
        
    def instruct_tts(self, text, prompt_wav_path, instruct_text, output_path, speed=1.0):
        """指令控制语音合成"""
        if not Path(prompt_wav_path).exists():
            raise FileNotFoundError(f"参考音频文件不存在: {prompt_wav_path}")
            
        prompt_speech = load_wav(str(prompt_wav_path), 16000)
        generator = self.model.inference_instruct2(
            tts_text=text,
            instruct_text=instruct_text,
            prompt_speech_16k=prompt_speech,
            stream=True,
            speed=speed
        )
        return self.stream_save_audio(generator, output_path)

    def synthesize_and_play(self, text, prompt_wav_path, output_path):
        """合成完整音频并播放"""
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