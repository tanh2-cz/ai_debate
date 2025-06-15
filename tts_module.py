"""
TTS模块 - 基于SiliconCloud API的文本转语音功能
为多角色辩论系统提供自动语音播放，包含音频长度获取
"""

import os
import requests
import base64
import time
import io
from typing import Dict, Optional, Tuple
from dotenv import find_dotenv, load_dotenv

# 音频处理库
try:
    from pydub import AudioSegment
    from pydub.utils import mediainfo
    AUDIO_PROCESSING_AVAILABLE = True
    print("✅ 音频处理库可用")
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    print("⚠️ 音频处理库不可用，将使用估算方法")

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
    
    def get_audio_duration(self, audio_bytes: bytes) -> float:
        """
        获取音频时长（秒）
        
        Args:
            audio_bytes: 音频数据字节
            
        Returns:
            音频时长（秒）
        """
        if not AUDIO_PROCESSING_AVAILABLE:
            # 备用估算方法：根据文件大小估算
            # MP3大约每秒128kbps = 16KB，但这很不准确
            estimated_duration = max(3, len(audio_bytes) / 16000)
            return estimated_duration
        
        try:
            # 使用pydub获取准确音频长度
            audio_io = io.BytesIO(audio_bytes)
            audio_segment = AudioSegment.from_mp3(audio_io)
            duration_seconds = len(audio_segment) / 1000.0  # pydub返回毫秒
            
            print(f"🎵 音频实际时长: {duration_seconds:.2f}秒")
            return duration_seconds
            
        except Exception as e:
            print(f"⚠️ 音频时长获取失败，使用估算: {e}")
            # 备用估算方法
            estimated_duration = max(3, len(audio_bytes) / 16000)
            return estimated_duration
    
    def text_to_speech(self, text: str, agent_role: str = "") -> Optional[Tuple[str, float]]:
        """
        将文本转换为语音
        
        Args:
            text: 要转换的文本
            agent_role: 角色标识，用于选择声音
            
        Returns:
            Tuple[base64编码的音频数据, 音频时长（秒）] 或 None
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
                "stream": False,
                "speed": 1.4,
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
            
            # 获取音频数据
            audio_bytes = response.content
            
            # 获取音频时长
            duration = self.get_audio_duration(audio_bytes)
            
            # 将音频数据转换为base64
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            print(f"✅ 语音生成成功: {len(audio_bytes)} bytes, {duration:.2f}秒")
            
            return (audio_base64, duration)
            
        except requests.exceptions.Timeout:
            print("❌ TTS API 请求超时")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ TTS API 请求错误: {e}")
            return None
        except Exception as e:
            print(f"❌ TTS 生成失败: {e}")
            return None
    
    def text_to_speech_simple(self, text: str, agent_role: str = "") -> Optional[str]:
        """
        简化版本，只返回base64音频数据（保持向后兼容）
        
        Args:
            text: 要转换的文本
            agent_role: 角色标识，用于选择声音
            
        Returns:
            base64编码的音频数据 或 None
        """
        result = self.text_to_speech(text, agent_role)
        if result:
            return result[0]  # 只返回音频数据，不返回时长
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
        
        # 测试带时长的接口
        test_text = "你好，这是一个测试文本。"
        test_role = "tech_expert"
        
        print(f"🔊 测试文本转语音（含时长）：{test_text}")
        result = tts.text_to_speech(test_text, test_role)
        
        if result:
            audio_data, duration = result
            print(f"✅ 测试成功：生成了 {len(audio_data)} 字符的base64音频数据")
            print(f"🎵 音频时长：{duration:.2f}秒")
        else:
            print("⚠️ 未生成音频数据")
        
        # 测试简化接口
        print(f"🔊 测试简化接口：{test_text}")
        simple_result = tts.text_to_speech_simple(test_text, test_role)
        
        if simple_result:
            print(f"✅ 简化接口测试成功：{len(simple_result)} 字符")
        else:
            print("⚠️ 简化接口测试失败")
            
    except Exception as e:
        print(f"❌ TTS模块测试失败: {e}")

if __name__ == "__main__":
    test_tts_module()