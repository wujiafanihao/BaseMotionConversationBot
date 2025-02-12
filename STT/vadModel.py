from modelscope import pipeline
from dotenv.main import load_dotenv
import os
import pyaudio
from wave import *
import keyboard
import asyncio
import time

class VADModel:
    def __init__(self):
        # 加载 .env 文件中的环境变量
        load_dotenv()
        # 设置说话人音频文件路径
        self.speaker_path = "STT/test/speaker.wav"

        self.model = pipeline(
            task='speaker-verification',
            model='iic/speech_campplus_sv_zh_en_16k-common_advanced',
            model_revision='v1.0.0',
            device='cuda'
        )

    async def initialize(self):
        """异步初始化方法"""
        await self._check_speaker_audio()

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

    async def _check_speaker_audio(self):
        """异步检查是否需要录制说话人音频"""
        if os.path.exists(self.speaker_path):
            while True:
                choice = input("\n检测到已存在说话人音频，是否需要重新录制？(y/n): ").lower()
                if choice in ['y', 'n']:
                    if choice == 'y':
                        print("\n即将开始重新录制说话人音频...")
                        await self._record_speaker_audio(self.speaker_path)
                    else:
                        print("\n将继续使用已有的说话人音频进行验证。")
                    break
                else:
                    print("\n请输入 y 或 n")
        else:
            print("\n未检测到说话人音频，即将开始录制...")
            # 确保目录存在
            os.makedirs(os.path.dirname(self.speaker_path), exist_ok=True)
            await self._record_speaker_audio(self.speaker_path)

    async def predict(self, audio_path):
        """
        异步预测方法
        将传入的音频与基准说话人音频(speaker.wav)进行对比
        """
        try:
            # 使用 asyncio.to_thread 将同步的模型推理转换为异步操作
            result = await asyncio.to_thread(
                self.model, 
                [self.speaker_path, audio_path], 
                thr=0.3
            )
            
            # 获取相似度得分
            similarity_score = result['score']
            # 获取是否为同一人的判定结果
            is_same_person = result['text'] == 'yes'
            
            # 构建返回消息
            message = f"相似度得分: {similarity_score:.4f}"
            if is_same_person:
                if similarity_score > 0.7:
                    message += " (✓ 非常确定是同一人)"
                elif similarity_score > 0.5:
                    message += " (✓ 很可能是同一人)"
                else:
                    message += " (✓ 可能是同一人)"
            else:
                if similarity_score < 0.2:
                    message += " (✗ 非常确定不是同一人)"
                elif similarity_score < 0.3:
                    message += " (✗ 很可能不是同一人)"
                else:
                    message += " (✗ 可能不是同一人)"
                
            return {
                'score': similarity_score,
                'is_same': is_same_person,
                'message': message
            }
        except Exception as e:
            print(f"程序运行出错: {str(e)}")
            return None

async def main():
    try:
        # 初始化模型
        vad = VADModel()
        await vad.initialize()
        
        print("\n开始说话人声音持续检测...")
        print("按 Ctrl+C 退出程序")
        
        # 初始化音频设置
        CHUNK = 1024
        RATE = 16000
        BUFFER_SECONDS = 3
        BUFFER_SIZE = int(RATE * BUFFER_SECONDS / CHUNK) * CHUNK

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        frames = []
        print("\n开始录音...")
        
        while True:
            # 读取音频数据
            data = stream.read(CHUNK)
            frames.append(data)
            
            # 保持缓冲区大小为3秒
            if len(frames) > BUFFER_SIZE // CHUNK:
                frames.pop(0)
            
            # 当累积够3秒数据时进行验证
            if len(frames) == BUFFER_SIZE // CHUNK:
                # 保存临时音频文件
                temp_file = "STT/test/temp_verify.wav"
                wf = Wave_write(temp_file)
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                # 异步进行验证
                result = await vad.predict(temp_file)
                if result:
                    print('\r' + result['message'], end='', flush=True)
                
                # 删除临时文件
                try:
                    os.remove(temp_file)
                except:
                    pass
                
                # 清空一部分缓冲区，保留部分重叠以实现平滑过渡
                frames = frames[len(frames)//2:]
            
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

if __name__ == '__main__':
    # 运行异步主函数
    asyncio.run(main())

