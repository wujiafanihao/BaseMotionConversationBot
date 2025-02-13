from modelscope import pipeline
from modelscope.utils.constant import Tasks
from dotenv.main import load_dotenv
import os
import pyaudio
from wave import *
import keyboard
import asyncio
from funasr import AutoModel
import warnings
warnings.filterwarnings("ignore")
import soundfile as sf
import numpy as np

class VADModel:
    def __init__(self):
        # 加载 .env 文件中的环境变量
        load_dotenv()
        
        # 设置说话人音频文件目录
        self.speaker_dir = "STT/test"
        os.makedirs(self.speaker_dir, exist_ok=True)
        
        # 音频配置
        self.SAMPLE_RATE = 16000
        self.CHANNELS = 1  # 修正为1通道
        self.CHUNK_SIZE_MS = 200  # 每个音频块的大小（毫秒）
        self.BIT_DEPTH = 16
        
        # VAD配置
        self.MIN_SPEECH_DURATION_MS = 500  # 最小语音片段长度（毫秒）
        self.MAX_MERGE_DISTANCE_MS = 300   # 最大合并距离（毫秒）
        self.SPEECH_PAD_MS = 100           # 语音片段前后填充（毫秒）
        
        # STT配置
        self.chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms
        self.encoder_chunk_look_back = 4  # encoder自注意力回看的块数
        self.decoder_chunk_look_back = 1  # decoder交叉注意力回看的块数
        self.chunk_stride = self.chunk_size[1] * 960  # 600ms
        
        # 注册的说话人信息
        self.reg_spks = {}
        self.current_speaker = None
        
        # 相似度阈值
        self.sv_threshold = 0.3
        
        # 说话人验证模型
        self.sv_pipeline = pipeline(
            task='speaker-verification',
            model='iic/speech_campplus_sv_zh_en_16k-common_advanced',
            model_revision='v1.0.0',
            device='cpu',
            window_size=400
        )

        # 语音活动检测模型
        self.vadModel = AutoModel(
            model="fsmn-vad",
            model_revision="v2.0.4",
            disable_pbar=True,
            max_end_silence_time=200,
            speech_noise_thres=0.8,
            disable_update=True
        )
        
        # 语音识别模型
        self.sttModel = AutoModel(
            model="paraformer-zh-streaming",
            model_revision="v2.0.4",
            disable_pbar=True
        )

    async def initialize(self):
        """异步初始化方法"""
        await self._load_registered_speakers()
        await self._select_speaker()

    async def _load_registered_speakers(self):
        """加载所有注册的说话人音频"""
        print("\n正在加载注册的说话人音频...")
        for file in os.listdir(self.speaker_dir):
            if file.endswith('.wav'):
                speaker_name = os.path.splitext(file)[0]
                file_path = os.path.join(self.speaker_dir, file)
                try:
                    data, sr = sf.read(file_path, dtype="float32")
                    self.reg_spks[speaker_name] = {
                        "data": data,
                        "sr": sr,
                        "path": file_path
                    }
                    print(f"已加载说话人: {speaker_name}")
                except Exception as e:
                    print(f"加载说话人 {speaker_name} 失败: {str(e)}")
        
        if not self.reg_spks:
            print("\n未检测到已注册的说话人，请先注册说话人。")
            await self._register_new_speaker()
        else:
            print(f"\n已成功加载 {len(self.reg_spks)} 个说话人。")

    async def _register_new_speaker(self):
        """注册新说话人"""
        speaker_name = input("\n请输入说话人姓名: ").strip()
        if not speaker_name:
            print("说话人姓名不能为空！")
            return
        
        file_path = os.path.join(self.speaker_dir, f"{speaker_name}.wav")
        await self._record_speaker_audio(file_path)
        
        try:
            data, sr = sf.read(file_path, dtype="float32")
            self.reg_spks[speaker_name] = {
                "data": data,
                "sr": sr,
                "path": file_path
            }
            print(f"\n说话人 {speaker_name} 注册成功！")
        except Exception as e:
            print(f"注册说话人失败: {str(e)}")
            if os.path.exists(file_path):
                os.remove(file_path)

    async def verify_speaker(self, audio_data):
        """
        验证说话人身份
        返回: (是否验证通过, 说话人姓名, 相似度得分)
        """
        if not self.reg_spks:
            print("没有注册的说话人信息！")
            return False, None, 0
            
        if not self.current_speaker:
            print("未选择目标说话人！")
            return False, None, 0
            
        # 保存临时音频文件用于验证
        temp_file = os.path.join(self.speaker_dir, "temp_verify.wav")
        try:
            wf = Wave_write(temp_file)
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_data)
            wf.close()
            
            if not os.path.exists(temp_file):
                print(f"临时文件创建失败: {temp_file}")
                return False, None, 0
            
            # 只验证当前选择的说话人
            speaker_info = self.reg_spks.get(self.current_speaker)
            if not speaker_info:
                print(f"目标说话人 {self.current_speaker} 的信息不存在")
                return False, None, 0
                
            if not os.path.exists(speaker_info["path"]):
                print(f"目标说话人音频文件不存在: {speaker_info['path']}")
                return False, None, 0
                    
            print(f"\n正在验证说话人: {self.current_speaker}")
            print(f"使用音频文件: {speaker_info['path']}")
                
            result = await asyncio.to_thread(
                self.sv_pipeline,
                [speaker_info["path"], temp_file],
                thr=self.sv_threshold
            )
                
            if isinstance(result, dict):
                score = float(result.get('score', 0))
                print(f"验证得分: {score:.4f} (阈值: {self.sv_threshold})")
                
                # 判断是否验证通过
                is_verified = score >= self.sv_threshold
                if is_verified:
                    print(f"\n✓ 验证通过！说话人: {self.current_speaker} (得分: {score:.4f})")
                else:
                    print(f"\n✗ 验证失败。得分: {score:.4f}, 阈值: {self.sv_threshold}")
                
                return is_verified, self.current_speaker, score
            else:
                print(f"验证结果格式错误: {result}")
                return False, None, 0
                    
        except Exception as e:
            print(f"说话人验证出错: {str(e)}")
            return False, None, 0
        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    print(f"清理临时文件失败: {str(e)}")

    async def _record_speaker_audio(self, filename):
        """
        异步录音方法
        按住Q键进行录音的函数，松开结束录制
        用于录制说话人基准音频
        """
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        
        frames = []
        print("\n请按住Q键开始录音，松开Q键结束录音...")
        
        # 等待按下Q键
        while not keyboard.is_pressed('q'):
            await asyncio.sleep(0.1)  # 非阻塞等待
        print("\n开始录音...")
        
        # 当Q键被按住时持续录音
        while keyboard.is_pressed('q'):
            data = stream.read(1024)
            frames.append(data)
            await asyncio.sleep(0.001)  # 给其他任务执行的机会
        
        print("录音结束.")
        
        # 关闭流
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # 保存录音到文件
        wf = Wave_write(filename)
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"音频已保存到: {filename}")

    async def _select_speaker(self):
        """选择要使用的说话人音频"""
        if not self.reg_spks:
            print("\n没有注册的说话人，请先注册。")
            await self._register_new_speaker()
            self.current_speaker = list(self.reg_spks.keys())[0]
            return

        print("\n已注册的说话人列表:")
        speakers = list(self.reg_spks.keys())
        for i, name in enumerate(speakers, 1):
            print(f"{i}. {name}")

        while True:
            try:
                choice = int(input("\n请选择要使用的说话人 (输入序号): "))
                if 1 <= choice <= len(speakers):
                    self.current_speaker = speakers[choice - 1]
                    print(f"\n已选择说话人: {self.current_speaker}")
                    break
                else:
                    print("无效的选择，请重试。")
            except ValueError:
                print("请输入有效的数字。")

    def _merge_segments(self, segments):
        """
        合并相近的语音片段
        Args:
            segments: 语音片段列表 [[start1, end1], [start2, end2], ...]
        Returns:
            list: 合并后的语音片段列表
        """
        if not segments:
            return []
            
        # 按开始时间排序
        segments = sorted(segments, key=lambda x: x[0])
        merged = []
        current = list(segments[0])
        
        for segment in segments[1:]:
            if segment[0] - current[1] <= self.MAX_MERGE_DISTANCE_MS:
                # 如果两个片段足够近，合并它们
                current[1] = segment[1]
            else:
                # 添加填充时间
                current[0] = max(0, current[0] - self.SPEECH_PAD_MS)
                current[1] = current[1] + self.SPEECH_PAD_MS
                # 如果片段长度足够长，保存它
                if current[1] - current[0] >= self.MIN_SPEECH_DURATION_MS:
                    merged.append(current)
                current = list(segment)
        
        # 处理最后一个片段
        current[0] = max(0, current[0] - self.SPEECH_PAD_MS)
        current[1] = current[1] + self.SPEECH_PAD_MS
        if current[1] - current[0] >= self.MIN_SPEECH_DURATION_MS:
            merged.append(current)
        
        return merged

    async def detect_voice_activity(self, audio_data):
        """
        使用VAD模型检测音频中的语音活动和端点
        Args:
            audio_data: 音频数据（字节流）
        Returns:
            tuple: (是否有语音, 语音片段列表)
        """
        try:
            # 将音频数据保存为临时文件供VAD模型使用
            temp_file = os.path.join(self.speaker_dir, "temp_vad.wav")
            wf = Wave_write(temp_file)
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(16000)
            wf.writeframes(audio_data)
            wf.close()

            if not os.path.exists(temp_file):
                print(f"VAD临时文件创建失败: {temp_file}")
                return False, []

            # 使用VAD模型进行检测
            result = self.vadModel.generate(temp_file)
            
            # 清理临时文件
            try:
                os.remove(temp_file)
            except Exception as e:
                print(f"清理VAD临时文件失败: {str(e)}")

            # 解析VAD结果
            if isinstance(result, list) and len(result) > 0:
                for item in result:
                    if isinstance(item, dict) and 'value' in item:
                        segments = item['value']
                        if segments and len(segments) > 0:
                            # 合并相近的语音片段
                            merged_segments = self._merge_segments(segments)
                            if merged_segments:
                                print(f"检测到语音片段: {merged_segments}")
                                return True, merged_segments
                return False, []
            return False, []

        except Exception as e:
            print(f"VAD检测出错: {str(e)}")
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            return False, []

    async def recognize_speech(self, audio_data):
        """
        对音频数据进行语音识别
        Args:
            audio_data: numpy数组格式的音频数据
        Returns:
            str: 识别出的文本
        """
        try:
            # 计算音频块数
            total_chunk_num = int(len(audio_data - 1) / self.chunk_stride + 1)
            cache = {}
            text_result = []
            
            # 逐块处理音频
            for i in range(total_chunk_num):
                speech_chunk = audio_data[i * self.chunk_stride:(i + 1) * self.chunk_stride]
                is_final = i == total_chunk_num - 1
                
                res = self.sttModel.generate(
                    input=speech_chunk,
                    cache=cache,
                    is_final=is_final,
                    chunk_size=self.chunk_size,
                    encoder_chunk_look_back=self.encoder_chunk_look_back,
                    decoder_chunk_look_back=self.decoder_chunk_look_back
                )
                
                if isinstance(res, list) and len(res) > 0:
                    for item in res:
                        if isinstance(item, dict) and 'text' in item:
                            text_result.append(item['text'])
            
            return ''.join(text_result)
            
        except Exception as e:
            print(f"语音识别出错: {str(e)}")
            return ""

    async def continuous_monitor(self):
        """
        持续监听音频输入，按住Q键时进行录音和识别
        """
        try:
            if not self.current_speaker:
                print("未选择说话人！")
                return

            print("\n开始语音识别系统...")
            print(f"当前选择的说话人: {self.current_speaker}")
            print("按住 'Q' 键进行说话")
            print("按 'R' 键注册新说话人")
            print("按 'S' 键切换说话人")
            print("按 'D' 键切换调试模式")
            print("按 'Ctrl+C' 退出程序")
            
            # 初始化音频设置
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=1024
            )
            
            debug_mode = False
            
            while True:
                # 检查键盘输入
                if keyboard.is_pressed('r'):
                    stream.stop_stream()
                    await self._register_new_speaker()
                    stream.start_stream()
                    await asyncio.sleep(0.5)
                    continue
                
                elif keyboard.is_pressed('s'):
                    stream.stop_stream()
                    await self._select_speaker()
                    stream.start_stream()
                    await asyncio.sleep(0.5)
                    continue
                
                elif keyboard.is_pressed('d'):
                    debug_mode = not debug_mode
                    print(f"\n调试模式: {'开启' if debug_mode else '关闭'}")
                    await asyncio.sleep(0.5)
                    continue

                # 当按下Q键时开始录音
                if keyboard.is_pressed('q'):
                    print("\n开始录音...")
                    frames = []
                    audio_data = []
                    
                    # 持续录音直到松开Q键
                    while keyboard.is_pressed('q'):
                        data = stream.read(1024, exception_on_overflow=False)
                        frames.append(data)
                        audio_data.extend(np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)
                        await asyncio.sleep(0.001)
                    
                    print("录音结束，正在处理...")
                    
                    if len(frames) > 0:
                        # 转换为音频数据
                        audio_bytes = b''.join(frames)
                        
                        # 进行说话人验证
                        is_verified, speaker_name, similarity_score = await self.verify_speaker(audio_bytes)
                        
                        if debug_mode:
                            print(f"\n调试信息:")
                            print(f"- 音频长度: {len(audio_data)} 样本")
                            print(f"- 相似度得分: {similarity_score:.4f}")
                            print(f"- 目标说话人: {self.current_speaker}")
                            print(f"- 识别说话人: {speaker_name}")
                        
                        if is_verified and speaker_name == self.current_speaker:
                            print(f"\n✓ 验证通过 - 说话人: {speaker_name} (相似度: {similarity_score:.4f})")
                            
                            # 进行语音识别
                            recognized_text = await self.recognize_speech(np.array(audio_data))
                            if recognized_text:
                                print(f"识别文本: {recognized_text}")
                            else:
                                print("未能识别出文本")
                        else:
                            print(f"\n✗ 验证失败 - 不是目标说话人 (相似度: {similarity_score:.4f})")
                    
                    await asyncio.sleep(0.5)  # 防止重复触发
                
                # 给其他任务执行的机会
                await asyncio.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n\n程序已退出")
        except Exception as e:
            print(f"\n程序运行出错: {str(e)}")
        finally:
            # 清理资源
            stream.stop_stream()
            stream.close()
            p.terminate()

async def main():
    try:
        # 初始化模型
        vad = VADModel()
        await vad.initialize()
        
        # 启动持续监听
        await vad.continuous_monitor()
            
    except Exception as e:
        print(f"\n程序运行出错: {str(e)}")

if __name__ == '__main__':
    # 运行异步主函数
    asyncio.run(main())

