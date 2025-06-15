"""
TTS模块 - 基于SiliconCloud API的文本转语音功能
为多角色辩论系统提供自动语音播放
"""

import os
import requests
import base64
import time
from typing import Dict, Optional
from dotenv import find_dotenv, load_dotenv

# 加载环境变量
load_dotenv(find_dotenv())

class TTSModule:
    """基于SiliconCloud API的文本转语音模块"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("SILICONCLOUD_API_KEY")
        self.api_url = "https://api.siliconflow.cn/v1/audio/speech"
        self.model = "FunAudioLLM/CosyVoice2-0.5B"
        
        # 为不同角色分配不同声音
        self.voice_mapping = {
            "environmentalist": "FunAudioLLM/CosyVoice2-0.5B:alex",    # 清澈声音
            "economist": "FunAudioLLM/CosyVoice2-0.5B:david",          # 沉稳声音
            "policy_maker": "FunAudioLLM/CosyVoice2-0.5B:charles",     # 权威声音
            "tech_expert": "FunAudioLLM/CosyVoice2-0.5B:benjamin",     # 理性声音
            "sociologist": "FunAudioLLM/CosyVoice2-0.5B:anna",         # 温和声音
            "ethicist": "FunAudioLLM/CosyVoice2-0.5B:claire"           # 严谨声音
        }
        
        if not self.api_key:
            print("⚠️ 警告: SILICONCLOUD_API_KEY 环境变量未设置")
        else:
            print("✅ TTS模块初始化成功")
    
    def text_to_speech(self, text: str, agent_role: str = "") -> Optional[str]:
        """
        将文本转换为语音
        
        Args:
            text: 要转换的文本
            agent_role: 角色标识，用于选择声音
            
        Returns:
            base64编码的音频数据，可直接用于HTML audio标签
        """
        
        if not self.api_key:
            print("❌ TTS API Key 未配置")
            return None
        
        if not text or not text.strip():
            print("⚠️ 文本内容为空")
            return None
        
        try:
            # 清理文本，移除角色名前缀
            clean_text = self._clean_text(text)
            
            # 选择声音
            voice = self.voice_mapping.get(agent_role, "FunAudioLLM/CosyVoice2-0.5B:alex")
            
            # 构建请求
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "input": clean_text,
                "voice": voice,
                "response_format": "mp3",
                "sample_rate": 32000,
                "stream": False,  # 简化处理，不使用流式
                "speed": 1.0,
                "gain": 0
            }
            
            print(f"🔊 正在生成语音: {agent_role} - {clean_text[:30]}...")
            
            # 调用API
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            
            # 将音频数据转换为base64
            audio_base64 = base64.b64encode(response.content).decode('utf-8')
            print(f"✅ 语音生成成功: {len(response.content)} bytes")
            
            return audio_base64
            
        except requests.exceptions.Timeout:
            print("❌ TTS API 请求超时")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ TTS API 请求错误: {e}")
            return None
        except Exception as e:
            print(f"❌ TTS 生成失败: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """清理文本，移除不必要的内容"""
        try:
            # 移除角色名前缀（如"环保主义者: "）
            if ":" in text:
                parts = text.split(":", 1)
                if len(parts) == 2:
                    text = parts[1].strip()
            
            # 限制文本长度（API限制）
            if len(text) > 1000:
                text = text[:1000]
            
            return text.strip()
            
        except Exception as e:
            print(f"⚠️ 文本清理失败: {e}")
            return text

# 全局TTS实例
tts_module = None

def initialize_tts_module() -> TTSModule:
    """初始化TTS模块"""
    global tts_module
    try:
        tts_module = TTSModule()
        print("🔊 TTS模块已初始化")
        return tts_module
    except Exception as e:
        print(f"❌ TTS模块初始化失败: {e}")
        return None

def get_tts_module() -> Optional[TTSModule]:
    """获取TTS模块实例"""
    return tts_module

def test_tts_module():
    """测试TTS模块功能"""
    print("🧪 开始测试TTS模块...")
    
    # 检查环境变量
    if not os.getenv("SILICONCLOUD_API_KEY"):
        print("❌ 警告: SILICONCLOUD_API_KEY 环境变量未设置")
        print("请设置环境变量：export SILICONCLOUD_API_KEY=your_api_key")
        return
    
    try:
        # 初始化TTS模块
        tts = initialize_tts_module()
        
        if not tts:
            print("❌ TTS模块初始化失败")
            return
        
        # 简单测试
        test_text = "你好，这是一个测试文本。"
        test_role = "tech_expert"
        
        print(f"🔊 测试文本转语音：{test_text}")
        audio_data = tts.text_to_speech(test_text, test_role)
        
        if audio_data:
            print(f"✅ 测试成功：生成了 {len(audio_data)} 字符的base64音频数据")
        else:
            print("⚠️ 未生成音频数据")
            
    except Exception as e:
        print(f"❌ TTS模块测试失败: {e}")

if __name__ == "__main__":
    test_tts_module()