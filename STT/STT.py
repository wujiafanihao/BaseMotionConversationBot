from modelscope import pipeline
from modelscope.utils.constant import Tasks
from dotenv.main import load_dotenv
import os
import pyaudio
from wave import *
import keyboard
import asyncio
from funasr import AutoModel
import soundfile as sf
import numpy as np

class SpeechRecognitionSystem:
    def __init__(self):
        # 加载 .env 文件中的环境变量
        load_dotenv()
        
        # 音频系统配置
        self.audio_config = {
            'sample_rate': 16000,
            'channels': 1,
            'chunk_size_ms': 200,
            'bit_depth': 16,
            'frame_size': 1024
        }
        
        # 文件系统配置
        self.storage_config = {
            'base_dir': "STT/test",
            'temp_vad_file': "temp_vad.wav",
            'temp_verify_file': "temp_verify.wav"
        }
        os.makedirs(self.storage_config['base_dir'], exist_ok=True)
        
        # 语音活动检测(VAD)配置
        self.vad_config = {
            'min_speech_duration_ms': 500,
            'max_merge_distance_ms': 300,
            'speech_pad_ms': 100
        }
        
        # 语音识别(STT)配置
        self.stt_config = {
            'chunk_size': [0, 10, 5],
            'encoder_lookback': 4,
            'decoder_lookback': 1,
            'stride': 10 * 960  # chunk_size[1] * 960
        }
        
        # 说话人管理
        self.speaker_manager = {
            'registered_speakers': {},
            'current_speaker': None,
            'verification_threshold': 0.3
        }
        
        # 初始化模型
        self._initialize_models()

    def _initialize_models(self):
        """初始化所有需要的AI模型"""
        # 说话人验证模型
        self.speaker_verifier = pipeline(
            task='speaker-verification',
            model='iic/speech_campplus_sv_zh_en_16k-common_advanced',
            model_revision='v1.0.0',
            device='cpu',
            window_size=400
        )

        # 语音活动检测模型
        self.voice_detector = AutoModel(
            model="fsmn-vad",
            model_revision="v2.0.4",
            disable_pbar=True,
            max_end_silence_time=200,
            speech_noise_thres=0.8,
            disable_update=True
        )
        
        # 语音识别模型
        self.speech_recognizer = AutoModel(
            model="paraformer-zh-streaming",
            model_revision="v2.0.4",
            disable_pbar=True,
            disable_update=True
        )

    async def initialize(self):
        """系统初始化"""
        await self._load_registered_speakers()
        await self._select_speaker()

    async def _load_registered_speakers(self):
        """加载所有注册的说话人音频"""
        print("\n正在加载注册的说话人音频...")
        for audio_file in os.listdir(self.storage_config['base_dir']):
            if audio_file.endswith('.wav'):
                speaker_name = os.path.splitext(audio_file)[0]
                audio_path = os.path.join(self.storage_config['base_dir'], audio_file)
                try:
                    audio_data, sample_rate = sf.read(audio_path, dtype="float32")
                    self.speaker_manager['registered_speakers'][speaker_name] = {
                        "data": audio_data,
                        "sample_rate": sample_rate,
                        "file_path": audio_path
                    }
                    print(f"已加载说话人: {speaker_name}")
                except Exception as e:
                    print(f"加载说话人 {speaker_name} 失败: {str(e)}")
        
        if not self.speaker_manager['registered_speakers']:
            print("\n未检测到已注册的说话人，请先注册说话人。")
            await self._register_new_speaker()
        else:
            print(f"\n已成功加载 {len(self.speaker_manager['registered_speakers'])} 个说话人。")

    async def _register_new_speaker(self):
        """注册新说话人"""
        speaker_name = input("\n请输入说话人姓名: ").strip()
        if not speaker_name:
            print("说话人姓名不能为空！")
            return
        
        audio_path = os.path.join(self.storage_config['base_dir'], f"{speaker_name}.wav")
        await self._record_speaker_audio(audio_path)
        
        try:
            audio_data, sample_rate = sf.read(audio_path, dtype="float32")
            self.speaker_manager['registered_speakers'][speaker_name] = {
                "data": audio_data,
                "sample_rate": sample_rate,
                "file_path": audio_path
            }
            print(f"\n说话人 {speaker_name} 注册成功！")
        except Exception as e:
            print(f"注册说话人失败: {str(e)}")
            if os.path.exists(audio_path):
                os.remove(audio_path)

    async def verify_speaker(self, audio_data):
        """
        验证说话人身份
        返回: (是否验证通过, 说话人姓名, 相似度得分)
        """
        if not self.speaker_manager['registered_speakers']:
            print("没有注册的说话人信息！")
            return False, None, 0
            
        current_speaker = self.speaker_manager['current_speaker']
        if not current_speaker:
            print("未选择目标说话人！")
            return False, None, 0
            
        # 保存临时音频文件用于验证
        temp_verify_path = os.path.join(self.storage_config['base_dir'], 
                                      self.storage_config['temp_verify_file'])
        try:
            with Wave_write(temp_verify_path) as wf:
                wf.setnchannels(self.audio_config['channels'])
                wf.setsampwidth(self.audio_config['bit_depth'] // 8)
                wf.setframerate(self.audio_config['sample_rate'])
                wf.writeframes(audio_data)
            
            if not os.path.exists(temp_verify_path):
                print(f"临时文件创建失败: {temp_verify_path}")
                return False, None, 0
            
            # 获取当前说话人信息
            speaker_info = self.speaker_manager['registered_speakers'].get(current_speaker)
            if not speaker_info:
                print(f"目标说话人 {current_speaker} 的信息不存在")
                return False, None, 0
                
            speaker_audio_path = speaker_info["file_path"]
            if not os.path.exists(speaker_audio_path):
                print(f"目标说话人音频文件不存在: {speaker_audio_path}")
                return False, None, 0
                    
            print(f"\n正在验证说话人: {current_speaker}")
            print(f"使用音频文件: {speaker_audio_path}")
                
            verification_result = await asyncio.to_thread(
                self.speaker_verifier,
                [speaker_audio_path, temp_verify_path],
                thr=self.speaker_manager['verification_threshold']
            )
                
            if isinstance(verification_result, dict):
                similarity_score = float(verification_result.get('score', 0))
                print(f"验证得分: {similarity_score:.4f} (阈值: {self.speaker_manager['verification_threshold']})")
                
                # 判断是否验证通过
                is_verified = similarity_score >= self.speaker_manager['verification_threshold']
                if is_verified:
                    print(f"\n✓ 验证通过！说话人: {current_speaker} (得分: {similarity_score:.4f})")
                else:
                    print(f"\n✗ 验证失败。得分: {similarity_score:.4f}, 阈值: {self.speaker_manager['verification_threshold']}")
                
                return is_verified, current_speaker, similarity_score
            else:
                print(f"验证结果格式错误: {verification_result}")
                return False, None, 0
                    
        except Exception as e:
            print(f"说话人验证出错: {str(e)}")
            return False, None, 0
        finally:
            # 清理临时文件
            if os.path.exists(temp_verify_path):
                try:
                    os.remove(temp_verify_path)
                except Exception as e:
                    print(f"清理临时文件失败: {str(e)}")

    async def _record_speaker_audio(self, output_path):
        """
        异步录音方法
        按住Q键进行录音的函数，松开结束录制
        用于录制说话人基准音频
        """
        audio_stream = pyaudio.PyAudio()
        recording_stream = audio_stream.open(
            format=pyaudio.paInt16,
            channels=self.audio_config['channels'],
            rate=self.audio_config['sample_rate'],
            input=True,
            frames_per_buffer=self.audio_config['frame_size']
        )
        
        audio_frames = []
        print("\n请按住Q键开始录音，松开Q键结束录音...")
        
        # 等待按下Q键
        while not keyboard.is_pressed('q'):
            await asyncio.sleep(0.1)
        print("\n开始录音...")
        
        # 当Q键被按住时持续录音
        while keyboard.is_pressed('q'):
            frame_data = recording_stream.read(self.audio_config['frame_size'])
            audio_frames.append(frame_data)
            await asyncio.sleep(0.001)
        
        print("录音结束.")
        
        # 关闭流
        recording_stream.stop_stream()
        recording_stream.close()
        audio_stream.terminate()
        
        # 保存录音到文件
        with Wave_write(output_path) as wf:
            wf.setnchannels(self.audio_config['channels'])
            wf.setsampwidth(audio_stream.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.audio_config['sample_rate'])
            wf.writeframes(b''.join(audio_frames))
        
        print(f"音频已保存到: {output_path}")

    async def _select_speaker(self):
        """选择要使用的说话人音频"""
        if not self.speaker_manager['registered_speakers']:
            print("\n没有注册的说话人，请先注册。")
            await self._register_new_speaker()
            self.speaker_manager['current_speaker'] = list(self.speaker_manager['registered_speakers'].keys())[0]
            return

        print("\n已注册的说话人列表:")
        registered_speakers = list(self.speaker_manager['registered_speakers'].keys())
        for idx, name in enumerate(registered_speakers, 1):
            print(f"{idx}. {name}")

        while True:
            try:
                selection = int(input("\n请选择要使用的说话人 (输入序号): "))
                if 1 <= selection <= len(registered_speakers):
                    self.speaker_manager['current_speaker'] = registered_speakers[selection - 1]
                    print(f"\n已选择说话人: {self.speaker_manager['current_speaker']}")
                    break
                else:
                    print("无效的选择，请重试。")
            except ValueError:
                print("请输入有效的数字。")

    def _merge_speech_segments(self, segments):
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
        sorted_segments = sorted(segments, key=lambda x: x[0])
        merged_segments = []
        current_segment = list(sorted_segments[0])
        
        for segment in sorted_segments[1:]:
            if segment[0] - current_segment[1] <= self.vad_config['max_merge_distance_ms']:
                # 如果两个片段足够近，合并它们
                current_segment[1] = segment[1]
            else:
                # 添加填充时间
                current_segment[0] = max(0, current_segment[0] - self.vad_config['speech_pad_ms'])
                current_segment[1] = current_segment[1] + self.vad_config['speech_pad_ms']
                # 如果片段长度足够长，保存它
                if current_segment[1] - current_segment[0] >= self.vad_config['min_speech_duration_ms']:
                    merged_segments.append(current_segment)
                current_segment = list(segment)
        
        # 处理最后一个片段
        current_segment[0] = max(0, current_segment[0] - self.vad_config['speech_pad_ms'])
        current_segment[1] = current_segment[1] + self.vad_config['speech_pad_ms']
        if current_segment[1] - current_segment[0] >= self.vad_config['min_speech_duration_ms']:
            merged_segments.append(current_segment)
        
        return merged_segments

    async def detect_voice_activity(self, audio_data):
        """
        使用VAD模型检测音频中的语音活动和端点
        Args:
            audio_data: 音频数据（字节流）
        Returns:
            tuple: (是否有语音, 语音片段列表)
        """
        temp_vad_path = os.path.join(self.storage_config['base_dir'], 
                                    self.storage_config['temp_vad_file'])
        try:
            # 保存临时音频文件
            with Wave_write(temp_vad_path) as wf:
                wf.setnchannels(self.audio_config['channels'])
                wf.setsampwidth(self.audio_config['bit_depth'] // 8)
                wf.setframerate(self.audio_config['sample_rate'])
                wf.writeframes(audio_data)

            if not os.path.exists(temp_vad_path):
                print(f"VAD临时文件创建失败: {temp_vad_path}")
                return False, []

            # 使用VAD模型进行检测
            vad_result = self.voice_detector.generate(temp_vad_path)
            
            # 解析VAD结果
            if isinstance(vad_result, list) and len(vad_result) > 0:
                for segment_info in vad_result:
                    if isinstance(segment_info, dict) and 'value' in segment_info:
                        speech_segments = segment_info['value']
                        if speech_segments and len(speech_segments) > 0:
                            # 合并相近的语音片段
                            merged_segments = self._merge_speech_segments(speech_segments)
                            if merged_segments:
                                print(f"检测到语音片段: {merged_segments}")
                                return True, merged_segments
                return False, []
            return False, []

        except Exception as e:
            print(f"VAD检测出错: {str(e)}")
            return False, []
        finally:
            # 清理临时文件
            if os.path.exists(temp_vad_path):
                try:
                    os.remove(temp_vad_path)
                except:
                    pass

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
            total_chunks = int(len(audio_data - 1) / self.stt_config['stride'] + 1)
            recognition_cache = {}
            recognized_text_parts = []
            
            # 逐块处理音频
            for chunk_idx in range(total_chunks):
                chunk_start = chunk_idx * self.stt_config['stride']
                chunk_end = (chunk_idx + 1) * self.stt_config['stride']
                speech_chunk = audio_data[chunk_start:chunk_end]
                is_final_chunk = chunk_idx == total_chunks - 1
                
                recognition_result = self.speech_recognizer.generate(
                    input=speech_chunk,
                    cache=recognition_cache,
                    is_final=is_final_chunk,
                    chunk_size=self.stt_config['chunk_size'],
                    encoder_chunk_look_back=self.stt_config['encoder_lookback'],
                    decoder_chunk_look_back=self.stt_config['decoder_lookback']
                )
                
                if isinstance(recognition_result, list) and len(recognition_result) > 0:
                    for result_item in recognition_result:
                        if isinstance(result_item, dict) and 'text' in result_item:
                            recognized_text_parts.append(result_item['text'])
            
            return ''.join(recognized_text_parts)
            
        except Exception as e:
            print(f"语音识别出错: {str(e)}")
            return ""

    async def continuous_monitor(self):
        """
        持续监听音频输入，按住Q键时进行录音和识别
        """
        try:
            if not self.speaker_manager['current_speaker']:
                print("未选择说话人！")
                return

            print("\n开始语音识别系统...")
            print(f"当前选择的说话人: {self.speaker_manager['current_speaker']}")
            print("按住 'Q' 键进行说话")
            print("按 'R' 键注册新说话人")
            print("按 'S' 键切换说话人")
            print("按 'D' 键切换调试模式")
            print("按 'Ctrl+C' 退出程序")
            
            # 初始化音频设置
            audio_system = pyaudio.PyAudio()
            audio_stream = audio_system.open(
                format=pyaudio.paInt16,
                channels=self.audio_config['channels'],
                rate=self.audio_config['sample_rate'],
                input=True,
                frames_per_buffer=self.audio_config['frame_size']
            )
            
            debug_mode = False
            
            while True:
                # 检查键盘输入
                if keyboard.is_pressed('r'):
                    audio_stream.stop_stream()
                    await self._register_new_speaker()
                    audio_stream.start_stream()
                    await asyncio.sleep(0.5)
                    continue
                
                elif keyboard.is_pressed('s'):
                    audio_stream.stop_stream()
                    await self._select_speaker()
                    audio_stream.start_stream()
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
                    audio_frames = []
                    audio_samples = []
                    
                    # 持续录音直到松开Q键
                    while keyboard.is_pressed('q'):
                        frame_data = audio_stream.read(self.audio_config['frame_size'], 
                                                     exception_on_overflow=False)
                        audio_frames.append(frame_data)
                        audio_samples.extend(np.frombuffer(frame_data, dtype=np.int16).astype(np.float32) / 32768.0)
                        await asyncio.sleep(0.001)
                    
                    print("录音结束，正在处理...")
                    
                    if len(audio_frames) > 0:
                        # 转换为音频数据
                        audio_data = b''.join(audio_frames)
                        
                        # 进行说话人验证
                        is_verified, speaker_name, similarity_score = await self.verify_speaker(audio_data)
                        
                        if debug_mode:
                            print(f"\n调试信息:")
                            print(f"- 音频长度: {len(audio_samples)} 样本")
                            print(f"- 相似度得分: {similarity_score:.4f}")
                            print(f"- 目标说话人: {self.speaker_manager['current_speaker']}")
                            print(f"- 识别说话人: {speaker_name}")
                        
                        if is_verified and speaker_name == self.speaker_manager['current_speaker']:
                            print(f"\n✓ 验证通过 - 说话人: {speaker_name} (相似度: {similarity_score:.4f})")
                            
                            # 进行语音识别
                            recognized_text = await self.recognize_speech(np.array(audio_samples))
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
            audio_stream.stop_stream()
            audio_stream.close()
            audio_system.terminate()

# async def main():
#     try:
#         # 初始化系统
#         speech_system = SpeechRecognitionSystem()
#         await speech_system.initialize()
        
#         # 启动持续监听
#         await speech_system.continuous_monitor()
            
#     except Exception as e:
#         print(f"\n程序运行出错: {str(e)}")

# if __name__ == '__main__':
#     # 运行异步主函数
#     asyncio.run(main())

