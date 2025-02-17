import gradio as gr
import cv2
import numpy as np
import os
import yaml
from pathlib import Path
from FaceEmotionDetector import FaceEmotionDetector
from STT.STT import SpeechRecognitionSystem
from LLM.model import ModelAdapter
from TTS.TTS import TTSSystem
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

class EmotionalChatApp:
    def __init__(self):
        # 使用线程池进行并行初始化
        with ThreadPoolExecutor(max_workers=4) as executor:
            print("正在初始化各个模块...")
            futures = {
                'face_detector': executor.submit(self._init_face_detector),
                'stt_system': executor.submit(self._init_stt_system),
                'llm': executor.submit(self._init_llm),
                'tts_system': executor.submit(self._init_tts_system)
            }
            
            # 获取初始化结果
            self.face_detector = futures['face_detector'].result()
            self.stt_system = futures['stt_system'].result()
            self.llm = futures['llm'].result()
            self.tts_system = futures['tts_system'].result()
            print("所有模块初始化完成！")
        
        # 当前对话模式
        self.current_mode = "text"  # "text" 或 "voice"
        
        # 获取说话人音频文件列表
        self.speaker_files = self._get_speaker_files()
        
        # 情绪标签列表
        self.emotion_labels = [
            "Angry",    # 0
            "Disgust",  # 1
            "Fear",     # 2
            "Happy",    # 3
            "Sad",      # 4
            "Surprise", # 5
            "Neutral",  # 6
            "Contempt"  # 7
        ]
    
    def save_llm_config(self, provider, model_name, thinking_model):
        """保存LLM配置"""
        try:
            # 更新LLM配置
            self.llm.update_config(
                provider=provider,
                model_name=model_name,
                thinking_model=thinking_model
            )
            return "LLM模型已更新！"
        except Exception as e:
            return f"更新LLM模型失败: {str(e)}"
    
    def save_config(self, tts_model, use_fp16):
        """保存TTS配置"""
        try:
            self.tts_system = TTSSystem(
                model_name=tts_model,
                use_fp16=use_fp16
            )
            return "TTS模型已更新！"
        except Exception as e:
            return f"更新TTS模型失败: {str(e)}"

    def _init_face_detector(self):
        return FaceEmotionDetector()
    
    def _init_stt_system(self):
        return SpeechRecognitionSystem()
    
    def _init_llm(self):
        return ModelAdapter()
    
    def _init_tts_system(self):
        return TTSSystem(model_name="cosyvoice-300m")

    def _get_speaker_files(self):
        """获取STT/test目录下的所有.wav文件"""
        speaker_dir = Path("STT/test")
        if not speaker_dir.exists():
            os.makedirs(speaker_dir)
        return [f.name for f in speaker_dir.glob("*.wav")]

    def process_camera_feed(self, frame):
        """处理摄像头画面，返回带情绪标注的画面"""
        if frame is None:
            return None, None
            
        # 检测情绪
        results = self.face_detector.detect_emotions(frame)
        
        # 在画面上标注情绪
        for result in results:
            x, y, w, h = result['box']
            emotion = result['emotion']
            # 绘制半透明的矩形框
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # 添加带背景的文本框
            cv2.rectangle(overlay, (x, y-30), (x+len(emotion)*15, y), (0, 255, 0), -1)
            cv2.putText(overlay, emotion, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # 将半透明效果应用到原始帧
            alpha = 0.6
            frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
        
        return frame, results[0]['emotion'] if results else None

    def handle_text_input(self, text, emotion_text):
        """处理文本输入"""
        try:
            # 获取思考过程和回答
            response, thoughts = self.llm.generate(text, emotion_text)
            
            # 启动一个线程来处理语音合成和播放
            def synthesize_and_play():
                try:
                    output_path = Path("TTS/output_audio/response.wav")
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    prompt_path = Path("TTS/CosyVoice/asset/zero_shot_prompt.wav")
                    
                    # 使用跨语言合成方法
                    success = self.tts_system.cross_lingual_synthesis(
                        text=response,
                        prompt_wav_path=str(prompt_path),
                        output_path=str(output_path),
                        stream=True,
                        speed=1.0
                    )
                    
                    if success:
                        return str(output_path)
                    return None
                except Exception as e:
                    print(f"语音合成出错: {str(e)}")
                    return None
            
            # 创建并启动语音合成线程
            tts_thread = threading.Thread(target=synthesize_and_play)
            tts_thread.daemon = True  # 设置为守护线程
            tts_thread.start()
            
            # 先返回思考和回答文本
            return thoughts, response, None
        except Exception as e:
            print(f"处理文本输入时出错: {str(e)}")
            return "处理出错", "抱歉，处理您的输入时出现了问题", None

    def handle_voice_input(self, audio_path, emotion):
        """处理语音输入"""
        if audio_path is None:
            return "未检测到音频", "请先录制音频", None
            
        try:
            # 进行说话人验证和语音识别
            is_verified, speaker, score = self.stt_system.verify_speaker(audio_path)
            if not is_verified:
                return "说话人验证失败", "请确认您的身份", None
                
            # 处理语音输入（这里需要实现语音转文字的功能）
            text = "语音转文字的结果"  # 替换为实际的语音转文字结果
            
            # 生成回答并合成语音
            return self.handle_text_input(text, emotion)
        except Exception as e:
            print(f"处理语音输入时出错: {str(e)}")
            return "处理出错", "抱歉，处理您的语音时出现了问题", None

    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks() as interface:
            # 创建标签页
            with gr.Tabs() as tabs:
                # 主对话页面
                with gr.Tab("对话"):
                    # 存储当前可见状态
                    state = gr.State({
                        "text_visible": True,
                        "voice_visible": False,
                        "camera_visible": False,
                        "speaker_visible": False
                    })
                    
                    with gr.Row():
                        # 左半部分
                        with gr.Column(scale=1):
                            # 摄像头画布（初始隐藏）
                            camera_group = gr.Group(visible=False)
                            with camera_group:
                                camera = gr.Image(source="webcam", streaming=True, label="摄像头画面", visible=False)
                                emotion_label = gr.Textbox(label="检测到的情绪", interactive=False)
                                with gr.Row():
                                    camera_btn = gr.Button("启动摄像头")
                                    camera_stop_btn = gr.Button("关闭摄像头")
                            
                            # 添加摄像头启动事件
                            def toggle_camera():
                                return gr.Image(visible=True)
                            
                            camera_btn.click(toggle_camera, outputs=[camera])
                            # 模式选择按钮
                            with gr.Row():
                                text_btn = gr.Button("文本对话", variant="primary")
                                voice_btn = gr.Button("语音对话")
                                gemini_btn = gr.Button("Gemini对话(开发中)", interactive=False)
                            
                            # 说话人选择（初始隐藏）
                            speaker_group = gr.Group(visible=False)
                            with speaker_group:
                                speaker_dropdown = gr.Dropdown(
                                    choices=self.speaker_files,
                                    label="选择说话人音频",
                                    interactive=True
                                )
                            
                            # 文本输入区域（默认显示）
                            text_input_group = gr.Group(visible=True)
                            with text_input_group:
                                text_input = gr.Textbox(
                                    label="文本输入",
                                    placeholder="请输入您想说的话...",
                                    lines=3
                                )
                                emotion_input = gr.Dropdown(
                                    choices=self.emotion_labels,
                                    label="选择情绪",
                                    value="Neutral",
                                    interactive=True
                                )
                                send_btn = gr.Button("发送", variant="primary")
                            
                            # 语音输入区域（初始隐藏）
                            voice_input_group = gr.Group(visible=False)
                            with voice_input_group:
                                audio_input = gr.Audio(
                                    source="microphone",
                                    type="filepath",
                                    label="点击录音"
                                )
                        
                        # 右半部分
                        with gr.Column(scale=1):
                            # 输出区域
                            thinking_output = gr.Textbox(
                                label="AI的思考",
                                lines=10,
                                interactive=False,
                                show_label=True
                            )
                            response_output = gr.Textbox(
                                label="AI的回答",
                                lines=5,
                                interactive=False,
                                show_label=True
                            )
                            clear_btn = gr.Button("清除输出")
                            
                            # 语音回复区域
                            audio_output = gr.Audio(
                                label="语音回复",
                                interactive=False,
                                show_label=True,
                                visible=True
                            )
                    
                    # 事件处理
                    def switch_to_text(state):
                        state.update({
                            "text_visible": True,
                            "voice_visible": False,
                            "camera_visible": False,
                            "speaker_visible": False
                        })
                        return [
                            gr.Group(visible=True),  # text_input_group
                            gr.Group(visible=False),  # voice_input_group
                            gr.Group(visible=False),  # camera_group
                            gr.Group(visible=False),  # speaker_group
                            state
                        ]
                    
                    def switch_to_voice(state):
                        state.update({
                            "text_visible": False,
                            "voice_visible": True,
                            "camera_visible": True,
                            "speaker_visible": True
                        })
                        return [
                            gr.Group(visible=False),  # text_input_group
                            gr.Group(visible=True),   # voice_input_group
                            gr.Group(visible=True),   # camera_group
                            gr.Group(visible=True),   # speaker_group
                            state
                        ]
                    
                    def clear_outputs():
                        return {
                            thinking_output: "",
                            response_output: "",
                            audio_output: None
                        }
                    
                    # 绑定事件
                    text_btn.click(
                        switch_to_text,
                        inputs=[state],
                        outputs=[text_input_group, voice_input_group, camera_group, speaker_group, state]
                    )
                    
                    voice_btn.click(
                        switch_to_voice,
                        inputs=[state],
                        outputs=[text_input_group, voice_input_group, camera_group, speaker_group, state]
                    )
                    
                    clear_btn.click(clear_outputs)
                    
                    # 处理摄像头输入
                    camera.stream(
                        self.process_camera_feed,
                        inputs=[camera],
                        outputs=[camera, emotion_label],
                        show_progress=False
                    )
                    
                    # 处理文本输入
                    send_btn.click(
                        self.handle_text_input,
                        inputs=[text_input, emotion_input],
                        outputs=[thinking_output, response_output, audio_output]
                    )
                    
                    # 处理语音输入
                    audio_input.stop_recording(
                        self.handle_voice_input,
                        inputs=[audio_input, emotion_label],
                        outputs=[thinking_output, response_output, audio_output]
                    )
                    
                    # 每隔一段时间检查语音文件是否生成完成
                    def check_audio():
                        """检查音频文件是否生成完成"""
                        try:
                            output_path = Path("TTS/output_audio/response.wav")
                            if output_path.exists():
                                return gr.Audio(value=str(output_path))
                            return None
                        except Exception as e:
                            print(f"检查音频文件时出错: {str(e)}")
                            return None
                    
                    # 添加定时更新音频输出的事件
                    audio_output.change(
                        check_audio,
                        inputs=None,
                        outputs=[audio_output],
                        every=1.0  # 每秒检查一次
                    )
                    
                    # 关闭摄像头的事件处理函数
                    def stop_camera():
                        """关闭摄像头并清空画面"""
                        return gr.Image(visible=False), gr.Textbox(value="")
                    
                    # 绑定关闭事件
                    camera_stop_btn.click(
                        stop_camera,
                        outputs=[camera, emotion_label]
                    )
                # 设置页面
                with gr.Tab("设置"):
                    with gr.Column():
                        gr.Markdown("## LLM 模型设置")
                        # LLM聊天模型设置
                        with gr.Group():
                            gr.Markdown("### 聊天模型设置")
                            llm_provider = gr.Radio(
                                choices=["OpenAI", "Gemini"],
                                label="选择模型提供商",
                                value="OpenAI"
                            )
                            
                            # OpenAI模型选项
                            openai_models = gr.Dropdown(
                                choices=["gpt-4o-mini", "gpt-4o"],
                                label="OpenAI模型",
                                value="gpt-4o-mini",
                                visible=True
                            )
                            
                            # Gemini模型选项
                            gemini_models = gr.Dropdown(
                                choices=[
                                    "gemini-1.5-flash-latest",
                                    "gemini-2.0-pro-exp-02-05",
                                    "gemini-2.0-flash-thinking-exp-1219"
                                ],
                                label="Gemini模型",
                                value="gemini-1.5-flash-latest",
                                visible=False
                            )
                        
                        # 思考模型设置
                        with gr.Group():
                            gr.Markdown("### 思考模型设置")
                            thinking_model = gr.Dropdown(
                                choices=["deepseek-r1:1.5b", "deepseek-r1:8b"],
                                label="思考模型",
                                value="deepseek-r1:1.5b"
                            )
                        
                        # 保存LLM设置的按钮
                        llm_save_btn = gr.Button("更新LLM模型", variant="primary")
                        llm_status_output = gr.Textbox(label="LLM设置状态", interactive=False)
                        
                        # 添加模型切换事件处理
                        # 在 EmotionalChatApp 类中的设置页面部分
                        def update_model_visibility(provider):
                            """更新模型选择器的可见性"""
                            if provider == "OpenAI":
                                return {
                                    openai_models: gr.update(visible=True),
                                    gemini_models: gr.update(visible=False)
                                }
                            else:  # provider == "Gemini"
                                return {
                                    openai_models: gr.update(visible=False),
                                    gemini_models: gr.update(visible=True)
                                }
                        
                        # 保存LLM设置
                        def save_llm_settings(provider, openai_model, gemini_model, thinking_model):
                            """保存 LLM 设置"""
                            try:
                                # 根据当前选择的提供商确定使用哪个模型名称
                                selected_model = openai_model if provider == "OpenAI" else gemini_model
                                self.llm.update_config(provider, selected_model, thinking_model)
                                return f"成功更新为 {provider} 的 {selected_model} 模型"
                            except Exception as e:
                                return f"更新LLM设置失败：{str(e)}"
                        
                        # 绑定事件
                        llm_provider.change(
                            update_model_visibility,
                            inputs=[llm_provider],
                            outputs=[openai_models, gemini_models]
                        )
                        
                        llm_save_btn.click(
                            save_llm_settings,
                            inputs=[llm_provider, openai_models, gemini_models, thinking_model],
                            outputs=[llm_status_output]
                        )
                        
                        gr.Markdown("## TTS 模型设置")
                        # 获取可用的TTS模型列表
                        tts_models = [
                            f"{model['name']} - {model['description']}"
                            for model in TTSSystem.get_available_models()
                        ]
                        tts_model = gr.Dropdown(
                            label="TTS模型",
                            choices=tts_models,
                            value=f"{self.tts_system.current_model} - {TTSSystem.AVAILABLE_MODELS[self.tts_system.current_model]['description']}",
                        )
                        use_fp16 = gr.Checkbox(
                            label="使用FP16半精度",
                            value=False
                        )
                        
                        save_btn = gr.Button("更新TTS模型", variant="primary")
                        status_output = gr.Textbox(label="状态", interactive=False)
                        
                        # 保存设置的回调函数
                        def save_settings(tts_model, use_fp16):
                            # 从显示文本中提取模型名称
                            tts_model_name = tts_model.split(" - ")[0]
                            return self.save_config(tts_model_name, use_fp16)
                        
                        save_btn.click(
                            save_settings,
                            inputs=[tts_model, use_fp16],
                            outputs=[status_output]
                        )
        
        return interface

def main():
    app = EmotionalChatApp()
    interface = app.create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

if __name__ == "__main__":
    main()
