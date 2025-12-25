import subprocess
import logging

# 检测CUDA支持
def is_cuda_supported():
    """检测系统是否支持CUDA加速"""
    try:
        # 检查NVIDIA驱动是否安装
        cmd = ["nvidia-smi"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            logging.info("检测到NVIDIA GPU和驱动")
            
            # 检查FFmpeg是否支持CUDA
            cmd = ["ffmpeg", "-hwaccels"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if "cuda" in result.stdout.lower():
                logging.info("FFmpeg支持CUDA加速")
                return True
            else:
                logging.info("FFmpeg不支持CUDA加速")
                return False
        else:
            logging.info("未检测到NVIDIA GPU或驱动")
            return False
    except Exception as e:
        logging.info(f"CUDA检测失败: {e}")
        return False