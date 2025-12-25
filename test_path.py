import os
import sys

# 模拟audio_analyzer.py的路径处理逻辑
SCAN_PATH = None
OUTPUT_PATH = None
TEMP_PATH = None

def test_main():
    global SCAN_PATH, OUTPUT_PATH, TEMP_PATH
    
    # 初始化路径
    current_dir = os.getcwd()
    SCAN_PATH = current_dir
    OUTPUT_PATH = os.path.join(current_dir, "Output")
    TEMP_PATH = os.path.join(current_dir, "Temp")
    
    print(f"当前工作目录: {current_dir}")
    print(f"SCAN_PATH: {SCAN_PATH}")
    print(f"OUTPUT_PATH: {OUTPUT_PATH}")
    print(f"TEMP_PATH: {TEMP_PATH}")

if __name__ == "__main__":
    test_main()