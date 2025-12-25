import os
import sys
import subprocess
import json
import csv
import logging
import concurrent.futures
import hashlib
import zipfile
import tempfile
import re
from datetime import datetime
from pathlib import Path

# 导入CUDA检测功能
import cuda_detection

# 配置matplotlib字体，解决中文和负号显示问题
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 全局变量
SCAN_PATH = None
OUTPUT_PATH = None
TEMP_PATH = None
AUDIO_EXTENSIONS = [".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg"]
MAX_FFMPEG_CONCURRENT = 4
FFMPEG_SEMAPHORE = None
CUDA_SUPPORTED = False  # CUDA支持标志
USE_CUDA = False  # 是否使用CUDA加速
MAX_GPU_CONCURRENT = 2  # GPU并发处理数量，通常低于CPU并发数

# 创建必要的目录
def create_directories():
    """创建输出和临时目录"""
    for path in [OUTPUT_PATH, TEMP_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"创建目录: {path}")

# 清理临时文件
def cleanup_temp_files():
    """清理临时文件"""
    try:
        if os.path.exists(TEMP_PATH):
            for root, dirs, files in os.walk(TEMP_PATH, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(TEMP_PATH)
            logging.info(f"清理临时目录: {TEMP_PATH}")
    except Exception as e:
        logging.error(f"清理临时文件失败: {e}")

# 获取最佳线程数
def get_optimal_thread_count():
    """获取系统最佳线程数"""
    return os.cpu_count() or 4

# 执行ffprobe命令
def run_ffprobe(file_path, format_flag="-show_format", stream_flag="-show_streams"):
    """执行ffprobe命令获取音频文件信息"""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            format_flag,
            stream_flag,
            "-print_format", "json",
            file_path
        ]
        # 先以字节形式获取输出，再手动解码
        result = subprocess.run(cmd, capture_output=True, check=True, text=False)
        
        # 尝试多种编码解码
        for encoding in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
            try:
                output = result.stdout.decode(encoding)
                return json.loads(output)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        
        logging.error(f"无法解码ffprobe输出: {file_path}")
        return None
    except subprocess.CalledProcessError as e:
        logging.error(f"ffprobe命令执行失败: {e}")
        return None
    except Exception as e:
        logging.error(f"运行ffprobe失败: {e}")
        return None

# 执行ffmpeg命令
def run_ffmpeg(cmd):
    """执行ffmpeg命令"""
    try:
        # 如果支持CUDA，添加CUDA加速参数
        if USE_CUDA:
            # 在命令开头添加CUDA加速参数
            if "ffmpeg" in cmd[0]:
                cmd.insert(1, "-hwaccel")
                cmd.insert(2, "cuda")
                cmd.insert(3, "-hwaccel_output_format")
                cmd.insert(4, "cuda")
        
        # 使用capture_output=True捕获输出，但不使用text=True
        result = subprocess.run(cmd, capture_output=True, check=True)
        # 显式解码输出，处理编码问题
        return result.stdout.decode('utf-8', errors='ignore')
    except subprocess.CalledProcessError as e:
        # 处理命令执行失败的情况
        stderr = e.stderr.decode('utf-8', errors='ignore') if e.stderr else ""
        logging.error(f"ffmpeg命令执行失败: {e}, 错误输出: {stderr}")
        return ""
    except Exception as e:
        # 捕获所有其他异常
        logging.error(f"执行ffmpeg命令时发生错误: {e}")
        return ""

# 扫描音频文件
def scan_audio_files(directory):
    """扫描目录中的音频文件"""
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in AUDIO_EXTENSIONS:
                file_path = os.path.join(root, file)
                audio_files.append(file_path)
    logging.info(f"扫描到 {len(audio_files)} 个音频文件")
    return audio_files

# 获取音质等级
def get_audio_quality(sample_rate, bits_per_sample, format_name):
    """根据采样率、位深度和格式判定音质等级"""
    # 确保参数为数值类型
    try:
        sample_rate = int(sample_rate) if sample_rate else 0
        bits_per_sample = int(bits_per_sample) if bits_per_sample else 0
    except (ValueError, TypeError):
        sample_rate = 0
        bits_per_sample = 0
    
    if format_name in ["mp3", "aac", "ogg"]:
        if sample_rate >= 96000 and bits_per_sample >= 24:
            return "高码率有损"
        elif sample_rate >= 44100 and bits_per_sample >= 16:
            return "标准有损"
        else:
            return "低码率有损"
    else:  # 无损格式
        if sample_rate >= 192000 and bits_per_sample >= 24:
            return "超高解析度"
        elif sample_rate >= 96000 and bits_per_sample >= 24:
            return "高解析度"
        elif sample_rate >= 44100 and bits_per_sample >= 16:
            return "CD音质"
        else:
            return "低于CD音质"

# 检测伪HiRes
def detect_fake_hires(file_path, sample_rate, bits_per_sample):
    """检测伪HiRes音频文件"""
    # 确保参数为数值类型
    try:
        sample_rate = int(sample_rate) if sample_rate else 0
        bits_per_sample = int(bits_per_sample) if bits_per_sample else 0
    except (ValueError, TypeError):
        return "否"  # 类型转换失败，不是HiRes
    
    if sample_rate < 96000 or bits_per_sample < 24:
        return "否"
    
    try:
        # 方法1：使用ffmpeg的highpass过滤器分析高频能量
        # 比较原始信号和高频信号的能量比
        cmd = [
            "ffmpeg", "-i", file_path,
            "-filter_complex", "aformat=channel_layouts=mono,asplit=2[original][high];" \
            "[high]highpass=f=16000[highpass];" \
            "[original]astats=metadata=1:reset=1[original_stats];" \
            "[highpass]astats=metadata=1:reset=1[highpass_stats];" \
            "[original_stats][highpass_stats]amerge",
            "-f", "null", "-"
        ]
        
        # 使用run_ffmpeg函数执行，会自动添加CUDA参数
        output = run_ffmpeg(cmd)
        
        if output is None:
            return "未知"
        
        # 提取能量值
        original_rms_match = re.search(r"RMS level dB:.*?: (-?\d+\.\d+)", output)
        highpass_rms_match = None
        # 安全地查找最后一个RMS level dB
        if "RMS level dB:" in output:
            # 从最后一个RMS level dB开始搜索
            last_rms_pos = output.rfind("RMS level dB:")
            if last_rms_pos != -1:
                highpass_rms_match = re.search(r"RMS level dB:.*?: (-?\d+\.\d+)", output[last_rms_pos:])
        
        if original_rms_match and highpass_rms_match:
            original_rms = float(original_rms_match.group(1))
            highpass_rms = float(highpass_rms_match.group(1))
            
            # 计算高频能量与原始能量的比值（dB差值）
            # 如果高频能量比原始能量低太多（>40dB），可能是假HiRes
            energy_diff = original_rms - highpass_rms
            
            # 合理的HiRes音频应该有一定的高频能量
            # 通常假HiRes的高频能量会非常低
            if energy_diff < 40:
                return "否"  # 高频能量足够，不是假HiRes
            else:
                return "是"  # 高频能量过低，可能是假HiRes
        else:
            # 如果无法提取能量值，使用备用方法
            # 方法2：检查是否有高频内容
            cmd = [
                "ffmpeg", "-i", file_path,
                "-filter_complex", "aformat=channel_layouts=mono,highpass=f=16000,volumedetect",
                "-f", "null", "-"
            ]
            
            output = run_ffmpeg(cmd)
            
            if output:
                # 检查高频信号是否有可检测的音量
                if "mean_volume:" in output:
                    mean_volume_match = re.search(r"mean_volume: (-?\d+\.\d+) dB", output)
                    if mean_volume_match:
                        mean_volume = float(mean_volume_match.group(1))
                        # 如果高频信号的平均音量高于-80dB，说明有实际内容
                        if mean_volume > -80:
                            return "否"  # 有高频内容，不是假HiRes
                        else:
                            return "是"  # 高频内容过少，可能是假HiRes
        
        return "未知"  # 无法确定
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg命令执行失败: {e}")
        return "未知"
    except Exception as e:
        logging.error(f"检测伪HiRes失败: {file_path} - {e}")
        import traceback
        logging.error(traceback.format_exc())
        return "未知"

# 分析音频DR、LUFS、峰值和剪辑检测
def analyze_loudness(file_path):
    """分析音频的DR、LUFS、峰值和剪辑检测"""
    try:
        # 使用ffmpeg的ebur128过滤器分析音频
        cmd = [
            "ffmpeg", "-i", file_path,
            "-filter_complex", "ebur128=peak=true",
            "-f", "null", "-"
        ]
        
        # 使用run_ffmpeg函数执行，会自动添加CUDA参数
        output = run_ffmpeg(cmd)
        
        if output is None:
            return {
                "DR": "",
                "LUFS": "",
                "峰值": "",
                "剪辑检测": ""
            }
        
        # 解析结果
        
        # 提取DR值（动态范围）
        dr_match = re.search(r"DR:\s*(\d+)\.\d+", output)
        dr = dr_match.group(1) if dr_match else ""
        
        # 提取LUFS值
        lufs_match = re.search(r"I:\s*(-?\d+)\.\d+ LUFS", output)
        lufs = lufs_match.group(1) if lufs_match else ""
        
        # 提取峰值
        peak_match = re.search(r"Peak:\s*(-?\d+)\.\d+ dBFS", output)
        peak = peak_match.group(1) if peak_match else ""
        
        # 检测剪辑
        clip_match = re.search(r"Clipping:\s*(\d+) samples", output)
        clipping = "是" if clip_match else "否"
        
        return {
            "DR": dr,
            "LUFS": lufs,
            "峰值": peak,
            "剪辑检测": clipping
        }
    except Exception as e:
        logging.error(f"分析音频响度失败: {file_path} - {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "DR": "",
            "LUFS": "",
            "峰值": "",
            "剪辑检测": ""
        }

# 分析音频基础信息
def analyze_basic_info(file_path):
    """分析音频文件的基础信息"""
    info = run_ffprobe(file_path)
    if not info:
        return None
    
    try:
        format_info = info.get("format", {})
        stream_info = info.get("streams", [])[0] if info.get("streams") else {}
        
        # 获取采样率
        sample_rate = stream_info.get("sample_rate", "")
        if sample_rate:
            sample_rate = int(sample_rate)
        
        # 获取位深度
        bits_per_sample = stream_info.get("bits_per_sample", "")
        if not bits_per_sample:
            # 尝试获取bits_per_raw_sample（适用于FLAC等无损格式）
            bits_per_sample = stream_info.get("bits_per_raw_sample", "")
        if not bits_per_sample:
            # 尝试从codec_name获取位深度信息
            codec_name = stream_info.get("codec_name", "")
            if "pcm_s" in codec_name:
                bits_per_sample = int(codec_name.split("_")[1])
        if bits_per_sample:
            bits_per_sample = int(bits_per_sample)
        
        # 获取声道数
        channels = stream_info.get("channels", "")
        if channels:
            channels = int(channels)
        
        # 获取实际码率
        bit_rate = format_info.get("bit_rate", "")
        actual_bitrate_kbps = ""  # 默认空字符串
        if bit_rate:
            actual_bitrate_kbps = round(int(bit_rate) / 1000, 2)
        
        # 计算原始码率（未压缩）
        original_bitrate_kbps = ""  # 默认空字符串
        if sample_rate and bits_per_sample and channels:
            original_bitrate_kbps = round(sample_rate * bits_per_sample * channels / 1000, 2)
        
        # 计算压缩率
        compression_rate = ""  # 默认空字符串
        if actual_bitrate_kbps and original_bitrate_kbps and original_bitrate_kbps != 0:
            compression_rate = round((1 - actual_bitrate_kbps / original_bitrate_kbps) * 100, 2)
        
        # 获取格式名称
        format_name = format_info.get("format_name", "").lower()
        if "." in format_name:
            format_name = format_name.split(".")[0]
        
        # 判定音质等级
        quality = get_audio_quality(sample_rate, bits_per_sample, format_name)
        
        # 检测伪HiRes
        fake_hires = detect_fake_hires(file_path, sample_rate, bits_per_sample)
        
        # 分析DR、LUFS、峰值和剪辑检测
        loudness_info = analyze_loudness(file_path)
        
        return {
            "文件名": os.path.basename(file_path),
            "路径": file_path,
            "格式": os.path.splitext(file_path)[1].lower()[1:],  # 去除点号
            "采样率": sample_rate,
            "位深": bits_per_sample,
            "声道": channels,
            "实际码率kbps": actual_bitrate_kbps,
            "原始码率kbps": original_bitrate_kbps,
            "压缩率百分比": compression_rate,
            "音质等级": quality,
            "伪HiRes": fake_hires,
            "DR": loudness_info["DR"],
            "LUFS": loudness_info["LUFS"],
            "峰值": loudness_info["峰值"],
            "剪辑检测": loudness_info["剪辑检测"]
        }
    except Exception as e:
        logging.error(f"分析音频基础信息失败: {file_path} - {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

# 并行分析所有音频文件的基础信息
def analyze_all_basic_info(audio_files):
    """并行分析所有音频文件的基础信息"""
    results = []
    thread_count = get_optimal_thread_count()
    
    # 如果支持CUDA，调整线程数以实现CPU与GPU协同处理
    if USE_CUDA:
        # GPU处理部分任务，CPU处理剩余任务
        total_thread_count = thread_count + MAX_GPU_CONCURRENT
        logging.info(f"使用 {total_thread_count} 个线程（{thread_count} CPU + {MAX_GPU_CONCURRENT} GPU）并行分析音频文件基础信息")
    else:
        logging.info(f"使用 {thread_count} 个CPU线程并行分析音频文件基础信息")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        future_to_file = {executor.submit(analyze_basic_info, file): file for file in audio_files}
        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                logging.error(f"分析文件失败: {file} - {e}")
    
    logging.info(f"完成 {len(results)} 个音频文件的基础信息分析")
    return results

# 计算文件哈希值
def compute_file_hash(file_path, hash_algorithm="md5"):
    """计算文件的哈希值"""
    hash_func = hashlib.new(hash_algorithm)
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except Exception as e:
        logging.error(f"计算文件哈希失败: {file_path} - {e}")
        return None

# 并行计算所有文件的哈希值
def compute_all_hashes(results):
    """并行计算所有文件的哈希值"""
    thread_count = get_optimal_thread_count()
    
    logging.info(f"使用 {thread_count} 个线程并行计算文件哈希值")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        future_to_result = {executor.submit(compute_file_hash, r["路径"]): r for r in results}
        for future in concurrent.futures.as_completed(future_to_result):
            result = future_to_result[future]
            try:
                hash_value = future.result()
                result["哈希值"] = hash_value
            except Exception as e:
                logging.error(f"计算文件哈希失败: {result['文件名']} - {e}")
                result["哈希值"] = ""
    
    logging.info("完成所有文件的哈希值计算")
    return results

# 分组重复文件
def group_duplicates(results):
    """分组重复文件（包括完全重复和相似文件）"""
    # 按哈希值分组（完全重复）
    hash_groups = {}
    for result in results:
        hash_value = result.get("哈希值", "")
        if hash_value:
            if hash_value not in hash_groups:
                hash_groups[hash_value] = []
            hash_groups[hash_value].append(result)
    
    # 为完全重复文件添加组号
    group_id = 1
    for group in hash_groups.values():
        if len(group) > 1:
            for result in group:
                result["重复组"] = group_id
                result["重复类型"] = "完全重复（哈希值相同）"
            group_id += 1
        else:
            for result in group:
                result["重复组"] = ""
                result["重复类型"] = ""
    
    logging.info(f"找到 {group_id - 1} 组重复文件")
    return results

# 生成Excel报告
def generate_excel_report(results):
    """生成Excel报告"""
    try:
        import pandas as pd
        
        # 准备数据，不包含路径信息
        data = []
        for result in results:
            data.append({
                "文件名": result.get('文件名', ''),
                "格式": result.get('格式', ''),
                "采样率": result.get('采样率', ''),
                "位深": result.get('位深', ''),
                "声道": result.get('声道', ''),
                "实际码率kbps": result.get('实际码率kbps', ''),
                "原始码率kbps": result.get('原始码率kbps', ''),
                "压缩率百分比": result.get('压缩率百分比', ''),
                "音质等级": result.get('音质等级', ''),
                "伪HiRes": result.get('伪HiRes', ''),
                "DR": result.get('DR', ''),
                "LUFS": result.get('LUFS', ''),
                "峰值": result.get('峰值', ''),
                "剪辑检测": result.get('剪辑检测', ''),
                "重复组": result.get('重复组', ''),
                "重复类型": result.get('重复类型', '')
            })
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 保存Excel报告
        excel_path = os.path.join(OUTPUT_PATH, "AudioReport.xlsx")
        df.to_excel(excel_path, index=False, engine='openpyxl')
        
        logging.info(f"生成Excel报告: {excel_path}")
    except ImportError:
        logging.error("生成Excel报告失败: 需要安装pandas和openpyxl库")
    except Exception as e:
        logging.error(f"生成Excel报告失败: {e}")

# 生成统计图表
import base64
import io

def generate_charts(results):
    """生成统计图表"""
    charts = {}
    
    try:
        import matplotlib.pyplot as plt
        
        # 1. 音质等级分布饼图
        quality_counts = {}
        for result in results:
            quality = result.get('音质等级', '未知')
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        if quality_counts:
            plt.figure(figsize=(8, 6))
            plt.pie(quality_counts.values(), labels=quality_counts.keys(), autopct='%1.1f%%', startangle=140)
            plt.title('音质等级分布', fontsize=14)
            plt.axis('equal')
            
            # 转换为base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            charts['quality_distribution'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
        
        # 2. 采样率分布直方图
        sample_rates = []
        for result in results:
            sr = result.get('采样率', 0)
            if sr and isinstance(sr, int):
                sample_rates.append(sr)
        
        if sample_rates:
            plt.figure(figsize=(10, 6))
            plt.hist(sample_rates, bins=20, edgecolor='black', alpha=0.7)
            plt.title('采样率分布', fontsize=14)
            plt.xlabel('采样率 (Hz)', fontsize=12)
            plt.ylabel('文件数量', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # 转换为base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            charts['sample_rate_distribution'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
        
        # 3. 伪HiRes检测结果柱状图
        fake_hires_counts = {'是': 0, '否': 0, '未知': 0}
        for result in results:
            fake = result.get('伪HiRes', '未知')
            if fake in fake_hires_counts:
                fake_hires_counts[fake] += 1
        
        plt.figure(figsize=(8, 6))
        plt.bar(fake_hires_counts.keys(), fake_hires_counts.values(), color=['red', 'green', 'gray'])
        plt.title('伪HiRes检测结果', fontsize=14)
        plt.xlabel('检测结果', fontsize=12)
        plt.ylabel('文件数量', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 转换为base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        charts['fake_hires_distribution'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        logging.info("生成统计图表完成")
    except ImportError:
        logging.error("生成统计图表失败: 需要安装matplotlib库")
    except Exception as e:
        logging.error(f"生成统计图表失败: {e}")
    
    return charts

# 生成HTML报告
def generate_html_report(results):
    """生成HTML报告"""
    try:
        # 准备数据
        analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_count = len(results)
        
        # 统计重复文件组数
        duplicate_groups = set()
        for result in results:
            if result.get("重复组", ""):
                duplicate_groups.add(result["重复组"])
        duplicate_groups_count = len(duplicate_groups)
        
        # 生成统计图表
        charts = generate_charts(results)
        
        # 生成表格行
        table_rows = ""
        for result in results:
            row_html = f"<tr>"
            row_html += f"<td>{result.get('文件名', '')}</td>"
            row_html += f"<td>{result.get('格式', '')}</td>"
            row_html += f"<td>{result.get('采样率', '')}</td>"
            row_html += f"<td>{result.get('位深', '')}</td>"
            row_html += f"<td>{result.get('声道', '')}</td>"
            row_html += f"<td>{result.get('实际码率kbps', '')}</td>"
            row_html += f"<td>{result.get('原始码率kbps', '')}</td>"
            row_html += f"<td>{result.get('压缩率百分比', '')}</td>"
            row_html += f"<td>{result.get('音质等级', '')}</td>"
            row_html += f"<td>{result.get('伪HiRes', '')}</td>"
            row_html += f"<td>{result.get('DR', '')}</td>"
            row_html += f"<td>{result.get('LUFS', '')}</td>"
            row_html += f"<td>{result.get('峰值', '')}</td>"
            row_html += f"<td>{result.get('剪辑检测', '')}</td>"
            row_html += f"<td>{result.get('重复组', '')}</td>"
            row_html += f"<td>{result.get('重复类型', '')}</td>"
            row_html += f"</tr>"
            table_rows += row_html
        
        # 生成图表HTML
        charts_html = ""
        if charts:
            charts_html += "<div class='charts-container'>"
            
            if 'quality_distribution' in charts:
                charts_html += f"<div class='chart-item'>"
                charts_html += f"<h3>音质等级分布</h3>"
                charts_html += f"<img src='data:image/png;base64,{charts['quality_distribution']}' alt='音质等级分布' class='chart-image'>"
                charts_html += f"</div>"
            
            if 'sample_rate_distribution' in charts:
                charts_html += f"<div class='chart-item'>"
                charts_html += f"<h3>采样率分布</h3>"
                charts_html += f"<img src='data:image/png;base64,{charts['sample_rate_distribution']}' alt='采样率分布' class='chart-image'>"
                charts_html += f"</div>"
            
            if 'fake_hires_distribution' in charts:
                charts_html += f"<div class='chart-item'>"
                charts_html += f"<h3>伪HiRes检测结果</h3>"
                charts_html += f"<img src='data:image/png;base64,{charts['fake_hires_distribution']}' alt='伪HiRes检测结果' class='chart-image'>"
                charts_html += f"</div>"
            
            charts_html += "</div>"
        
        # 生成HTML内容
        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>音频质量分析报告</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 0;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }}
        
        header h1 {{
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 15px;
            text-align: center;
        }}
        
        .summary {{
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
        }}
        
        .summary-item {{
            background: rgba(255, 255, 255, 0.2);
            padding: 15px 25px;
            border-radius: 6px;
            backdrop-filter: blur(10px);
        }}
        
        .summary-item strong {{
            font-size: 18px;
        }}
        
        .charts-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .chart-item {{
            flex: 1;
            min-width: 300px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        }}
        
        .chart-item h3 {{
            text-align: center;
            margin-bottom: 15px;
            color: #555;
        }}
        
        .chart-image {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }}
        
        .filter-section {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        }}
        
        .filter-section h3 {{
            margin-bottom: 15px;
            color: #555;
        }}
        
        .filter-group {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        
        .filter-group label {{margin-right: 10px;font-weight: 500;}}
        .filter-group select {{padding: 8px 12px;border: 1px solid #ddd;border-radius: 4px;background: white;}}
        </style>
        <script>
            // 筛选功能
            function filterTable() {{
                // 音质等级筛选
                const qualityFilter = document.getElementById('qualityFilter');
                const qualityValue = qualityFilter.value;
                
                // 伪HiRes筛选
                const hiresFilter = document.getElementById('hiresFilter');
                const hiresValue = hiresFilter.value;
                
                // 表格行
                const rows = document.querySelectorAll('table tbody tr');
                
                rows.forEach(row => {{
                    const qualityCell = row.cells[8]; // 音质等级列
                    const hiresCell = row.cells[9];   // 伪HiRes列
                    
                    let showRow = true;
                    
                    // 音质等级筛选
                    if (qualityValue && qualityCell.textContent !== qualityValue) {{showRow = false;}}
                    
                    // 伪HiRes筛选
                    if (hiresValue && hiresCell.textContent !== hiresValue) {{showRow = false;}}
                    
                    row.style.display = showRow ? '' : 'none';
                }});
            }}
        </script>
</head>
<body>
    <div class="container">
        <header>
            <h1>音频质量分析报告</h1>
            <div class="summary">
                <div class="summary-item">
                    <strong>分析时间:</strong> {analysis_time}
                </div>
                <div class="summary-item">
                    <strong>分析文件数:</strong> {file_count}
                </div>
                <div class="summary-item">
                    <strong>重复文件组数:</strong> {duplicate_groups_count}
                </div>
            </div>
        </header>
        
        {charts_html}
        
        <div class="filter-section">
            <h3>筛选功能</h3>
            <div class="filter-group">
                <div>
                    <label for="qualityFilter">音质等级:</label>
                    <select id="qualityFilter" onchange="filterTable()">
                        <option value="">全部</option>
                        <option value="超高解析度">超高解析度</option>
                        <option value="高解析度">高解析度</option>
                        <option value="CD音质">CD音质</option>
                        <option value="低于CD音质">低于CD音质</option>
                        <option value="高码率有损">高码率有损</option>
                        <option value="标准有损">标准有损</option>
                        <option value="低码率有损">低码率有损</option>
                    </select>
                </div>
                <div>
                    <label for="hiresFilter">伪HiRes:</label>
                    <select id="hiresFilter" onchange="filterTable()">
                        <option value="">全部</option>
                        <option value="是">是</option>
                        <option value="否">否</option>
                        <option value="未知">未知</option>
                    </select>
                </div>
            </div>
        </div>
        
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>文件名</th><th>格式</th><th>采样率</th><th>位深</th>
                        <th>声道</th><th>实际码率kbps</th><th>原始码率kbps</th><th>压缩率百分比</th>
                        <th>音质等级</th><th>伪HiRes</th><th>DR</th><th>LUFS</th><th>峰值</th>
                        <th>剪辑检测</th><th>重复组</th><th>重复类型</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        
        <footer>
            <p>音频质量分析工具 v1.0 | 生成时间: {analysis_time}</p>
        </footer>
    </div>
</body>
</html>"""
        
        # 保存HTML报告
        html_path = os.path.join(OUTPUT_PATH, "AudioReport.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logging.info(f"生成HTML报告: {html_path}")
    except Exception as e:
        logging.error(f"生成HTML报告失败: {e}")

# 主函数
def main():
    """主函数"""
    global SCAN_PATH, OUTPUT_PATH, TEMP_PATH, CUDA_SUPPORTED, USE_CUDA
    try:
        logging.info("开始音频质量分析")
        
        # 检测CUDA支持情况
        logging.info("检测CUDA支持情况...")
        CUDA_SUPPORTED = cuda_detection.is_cuda_supported()
        if CUDA_SUPPORTED:
            USE_CUDA = True  # 默认使用CUDA加速
            logging.info("CUDA支持已启用，将使用GPU加速处理")
        else:
            logging.info("CUDA不支持，使用CPU处理")
        
        # 初始化路径
        current_dir = os.getcwd()
        SCAN_PATH = current_dir
        OUTPUT_PATH = os.path.join(current_dir, "Output")
        TEMP_PATH = os.path.join(current_dir, "Temp")
        
        # 初始化环境
        create_directories()
        
        # 扫描音频文件
        logging.info(f"扫描路径: {SCAN_PATH}")
        audio_files = scan_audio_files(SCAN_PATH)
        if not audio_files:
            logging.info("没有找到音频文件")
            return 0
        
        logging.info(f"找到 {len(audio_files)} 个音频文件")
        
        # 分析音频基础信息
        logging.info("开始分析音频基础信息")
        results = analyze_all_basic_info(audio_files)
        if not results:
            logging.info("音频文件分析失败")
            return 1
        
        # 计算文件哈希并检测重复
        logging.info("计算文件哈希并检测重复")
        compute_all_hashes(results)
        group_duplicates(results)
        
        # 生成HTML报告
        logging.info("生成HTML报告")
        generate_html_report(results)
        
        # 生成Excel报告
        logging.info("生成Excel报告")
        generate_excel_report(results)
        
        # 清理临时文件
        cleanup_temp_files()
        
        logging.info("音频分析完成")
        logging.info(f"分析结果保存在: {OUTPUT_PATH}")
        
        return 0
    except Exception as e:
        logging.error(f"程序执行失败: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())