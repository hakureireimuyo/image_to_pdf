import os
import argparse
from PIL import Image
import torch
import torchvision.transforms as transforms
from PyPDF2 import PdfMerger, PdfReader
from PyPDF2.errors import PdfReadError
from tqdm import tqdm
import concurrent.futures
from pathlib import Path
import re
import numpy as np
import shutil
import tempfile
from datetime import datetime
import io
import time
import zipfile
import py7zr
import rarfile
import mimetypes
import logging
import gc

# 预设纸张尺寸（单位：毫米）
PAPER_SIZES = {
    'A4': (210, 297),  # 宽, 高
    'A5': (148, 210),
    'A3': (297, 420),
    'B5': (176, 250),
    'B4': (250, 353),
    'Letter': (215.9, 279.4),
    'Legal': (215.9, 355.6)
}

# 配置日志
def setup_logger():
    """配置日志记录器"""
    # 创建logs目录（如果不存在）
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 生成日志文件名（使用当前时间）
    log_filename = f'logs/image_to_pdf_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # 配置日志记录器
    logger = logging.getLogger('image_to_pdf')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def cleanup_resources():
    """清理资源"""
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 强制进行垃圾回收
    gc.collect()

def print_usage():
    """打印使用说明"""
    print("""
图片转PDF工具使用说明
====================

基本用法：
    python image_to_pdf.py "输入路径"

完整参数：
    python image_to_pdf.py "输入路径" [选项]

参数说明：
    输入路径：
        可以是单个路径或多个路径（用分号分隔）
        例如：F:\\图片文件夹 或 F:\\图片1;F:\\图片2

选项：
    --output, -o      输出目录路径（可选）
    --size, -s        预设纸张尺寸，可选值：
                      A4, A5, A3, B5, B4, Letter, Legal
                      默认：A4
    --orientation     纸张方向，可选值：
                      portrait（竖版）
                      landscape（横版）
                      默认：portrait
    --ppi            图片分辨率（每英寸像素数）
                      默认：250
    --custom-size    启用自定义尺寸模式
                      需要同时指定--width和--height
    --width          自定义PDF页面宽度（像素）
                      需要与--custom-size一起使用
    --height         自定义PDF页面高度（像素）
                      需要与--custom-size一起使用
    --no-keep-original-ratio
                     不保持原图比例，将图片拉伸至指定尺寸
                     默认保持原图比例
    --keep-original-size
                     保持原图大小，忽略所有尺寸设置
    --use-memory     使用内存存储临时PDF文件
                     默认使用磁盘存储
    --no-gpu         强制使用CPU处理
                     默认使用GPU（如果可用）
    --sort-by        文件排序方式，可选值：
                     name（文件名）
                     created（创建时间）
                     modified（修改时间）
                     size（文件大小）
                     默认：name
    --show-files     显示排序后的文件列表
    --delete-source  转换成功后删除源文件/文件夹
    --help, -h       显示此帮助信息

示例：
1. 基本用法（不保持原图比例，按文件名排序）：
   python image_to_pdf.py "F:\\图片文件夹"

2. 使用自定义尺寸：
   python image_to_pdf.py "F:\\图片文件夹" --custom-size --width 1920 --height 1080

3. 保持原图比例：
   python image_to_pdf.py "F:\\图片文件夹" --no-keep-original-ratio

4. 保持原图大小：
   python image_to_pdf.py "F:\\图片文件夹" --keep-original-size

5. 使用内存存储：
   python image_to_pdf.py "F:\\图片文件夹" --use-memory

6. 转换后删除源文件：
   python image_to_pdf.py "F:\\图片文件夹" --delete-source

7. 使用横版A4：
   python image_to_pdf.py "F:\\图片文件夹" --orientation landscape

注意事项：
1. 默认情况下不保持原图比例，将图片缩放至指定尺寸
2. 使用--no-keep-original-ratio可以不保持原图比例，将图片拉伸至指定尺寸
3. 使用--keep-original-size可以保持原图大小，忽略所有尺寸设置
4. 使用--use-memory可以将临时PDF文件存储在内存中，提高处理速度
5. 程序会自动检测是否可以使用GPU，如果可用则使用GPU加速
6. 文件会按照指定的方式排序（默认按文件名）
7. 使用--show-files选项可以查看排序后的文件列表
8. 使用--delete-source选项可以在转换成功后删除源文件/文件夹
9. 使用--orientation选项可以指定纸张方向（横版或竖版）
""")

def natural_sort_key(s):
    """自然排序键函数，用于文件名排序"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))]

def get_sort_key(file_path, sort_by):
    """根据不同的排序方式获取排序键"""
    if sort_by == 'name':
        return natural_sort_key(file_path.name)
    elif sort_by == 'created':
        return os.path.getctime(file_path)
    elif sort_by == 'modified':
        return os.path.getmtime(file_path)
    elif sort_by == 'size':
        return os.path.getsize(file_path)
    else:
        return natural_sort_key(file_path.name)

def mm_to_pixels(mm, ppi):
    """将毫米转换为像素"""
    return int(mm * ppi / 25.4)

def get_device():
    """获取可用的设备（GPU或CPU）"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def is_valid_pdf(pdf_path):
    """验证PDF文件是否有效"""
    try:
        with open(pdf_path, 'rb') as f:
            PdfReader(f)
        return True
    except Exception:
        return False

def process_image_gpu(image_path, target_size, device, ppi, keep_original_ratio=True, keep_original_size=False, use_memory=False):
    """使用GPU处理单个图片"""
    try:
        # 读取图片
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if keep_original_size:
                # 保持原图大小
                output_img = img
            else:
                # 转换为PyTorch张量并移动到GPU
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
                # 确保张量维度顺序为 [1, 3, H, W]
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                # 计算调整后的尺寸
                img_width, img_height = img.size
                target_width, target_height = target_size
                
                if keep_original_ratio:
                    # 保持原图比例，计算缩放后的尺寸
                    width_ratio = target_width / img_width
                    height_ratio = target_height / img_height
                    scale_ratio = min(width_ratio, height_ratio)
                    new_width = int(img_width * scale_ratio)
                    new_height = int(img_height * scale_ratio)
                    output_size = (new_height, new_width)  # 注意：Resize需要(height, width)
                    
                    # 使用GPU进行图像缩放
                    resize_transform = transforms.Resize(output_size, antialias=True)
                    img_tensor = resize_transform(img_tensor)  # [1, 3, new_height, new_width]
                    
                    # 创建白色背景（在GPU上）
                    background = torch.ones((1, 3, target_height, target_width), device=device)
                    
                    # 计算居中位置
                    x = (target_width - output_size[1]) // 2  # 注意：output_size[1]是宽度
                    y = (target_height - output_size[0]) // 2  # 注意：output_size[0]是高度
                    
                    # 将调整后的图片粘贴到背景上（在GPU上）
                    background[:, :, y:y+output_size[0], x:x+output_size[1]] = img_tensor
                    img_tensor = background
                else:
                    # 不保持原图比例，直接缩放到目标尺寸
                    resize_transform = transforms.Resize((target_height, target_width), antialias=True)
                    img_tensor = resize_transform(img_tensor)
                
                # 转换回PIL图像（在CPU上）
                img_tensor = img_tensor.squeeze(0).cpu()  # [3, H, W]
                img_tensor = torch.clamp(img_tensor, 0, 1)
                img_array = (img_tensor.numpy() * 255).astype(np.uint8)
                img_array = np.transpose(img_array, (1, 2, 0))  # [H, W, 3]
                output_img = Image.fromarray(img_array)
            
            # 保存为PDF
            if use_memory:
                # 使用内存存储
                pdf_buffer = io.BytesIO()
                output_img.save(pdf_buffer, 'PDF', resolution=ppi)
                pdf_buffer.seek(0)
                return pdf_buffer
            else:
                # 使用临时文件存储
                temp_dir = tempfile.mkdtemp()
                temp_pdf = os.path.join(temp_dir, f"{os.path.basename(image_path)}_temp.pdf")
                output_img.save(temp_pdf, 'PDF', resolution=ppi)
                return temp_pdf
            
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {str(e)}")
        return None

def process_directory(input_path, output_path, target_size, device, ppi, keep_original_ratio=True, keep_original_size=False, use_memory=False, sort_by='name', show_file_list=False, memory_files=None, is_archive=False, logger=None):
    """处理目录中的所有图片"""
    start_time = time.time()
    logger.info(f"开始处理: {input_path}")
    
    try:
        # 支持的图片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        # 获取所有图片文件
        if memory_files:
            # 从内存文件系统中获取图片
            image_files = []
            for file_path, file_content in memory_files.items():
                if any(file_path.lower().endswith(ext) for ext in image_extensions):
                    # 创建临时文件
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_path).suffix)
                    temp_file.write(file_content)
                    temp_file.close()
                    # 保存原始文件名和临时文件路径
                    image_files.append({
                        'temp_path': Path(temp_file.name),
                        'original_name': Path(file_path).name,
                        'content': file_content
                    })
            
            # 根据排序方式对文件进行排序
            if sort_by == 'name':
                image_files.sort(key=lambda x: natural_sort_key(x['original_name']))
            elif sort_by == 'size':
                image_files.sort(key=lambda x: len(x['content']))
            elif sort_by in ['created', 'modified']:
                logger.warning(f"压缩包中的文件不支持按{sort_by}排序，将使用文件名排序")
                image_files.sort(key=lambda x: natural_sort_key(x['original_name']))
            
            # 提取临时文件路径列表
            image_files = [item['temp_path'] for item in image_files]
        else:
            # 从文件系统中获取图片
            image_files = set()
            for ext in image_extensions:
                image_files.update(Path(input_path).glob(f'*{ext}'))
                image_files.update(Path(input_path).glob(f'*{ext.upper()}'))
            image_files = list(image_files)
            
            # 排序
            image_files = sorted(image_files, key=lambda x: get_sort_key(x, sort_by))
        
        if not image_files:
            logger.warning(f"在 {input_path} 中没有找到支持的图片文件")
            return None
        
        logger.info(f"找到 {len(image_files)} 个图片文件")
        
        # 显示排序后的文件列表
        if show_file_list:
            logger.info("\n排序后的文件列表:")
            for i, file in enumerate(image_files, 1):
                if is_archive and memory_files:
                    file_name = file.name
                    if sort_by == 'size':
                        size_str = f"{len(memory_files[file_name]) / 1024:.1f} KB"
                        logger.info(f"{i}. {file_name} (大小: {size_str})")
                    else:
                        logger.info(f"{i}. {file_name}")
                else:
                    if sort_by == 'created':
                        time_str = datetime.fromtimestamp(os.path.getctime(file)).strftime('%Y-%m-%d %H:%M:%S')
                        logger.info(f"{i}. {file.name} (创建时间: {time_str})")
                    elif sort_by == 'modified':
                        time_str = datetime.fromtimestamp(os.path.getmtime(file)).strftime('%Y-%m-%d %H:%M:%S')
                        logger.info(f"{i}. {file.name} (修改时间: {time_str})")
                    elif sort_by == 'size':
                        size_str = f"{os.path.getsize(file) / 1024:.1f} KB"
                        logger.info(f"{i}. {file.name} (大小: {size_str})")
                    else:
                        logger.info(f"{i}. {file.name}")
        
        # 创建临时目录（如果不使用内存存储）
        temp_dir = None if use_memory else tempfile.mkdtemp()
        temp_pdfs = [None] * len(image_files)  # 预分配列表，保持顺序
        
        try:
            # 使用线程池并行处理图片
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # 创建任务字典，记录每个任务的索引
                future_to_index = {
                    executor.submit(
                        process_image_gpu, 
                        str(img), 
                        target_size, 
                        device, 
                        ppi, 
                        keep_original_ratio, 
                        keep_original_size, 
                        use_memory
                    ): i for i, img in enumerate(image_files)
                }
                
                # 处理完成的任务
                for future in tqdm(concurrent.futures.as_completed(future_to_index), 
                                  total=len(future_to_index), 
                                  desc="处理图片"):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        if result:
                            temp_pdfs[index] = result
                    except Exception as e:
                        logger.error(f"处理图片 {image_files[index]} 时出错: {str(e)}")
            
            # 过滤掉None值（处理失败的文件）
            temp_pdfs = [pdf for pdf in temp_pdfs if pdf is not None]
            
            if not temp_pdfs:
                logger.error("没有成功处理的图片")
                return None
            
            # 合并PDF
            merger = PdfMerger()
            processed_files = set()  # 用于记录已处理的文件
            
            for i, pdf in enumerate(temp_pdfs):
                try:
                    if use_memory:
                        # 检查是否已经处理过这个文件
                        file_name = image_files[i].name
                        if file_name not in processed_files:
                            merger.append(pdf)
                            processed_files.add(file_name)
                    else:
                        if is_valid_pdf(pdf):
                            # 检查是否已经处理过这个文件
                            file_name = image_files[i].name
                            if file_name not in processed_files:
                                merger.append(pdf)
                                processed_files.add(file_name)
                except PdfReadError as e:
                    logger.error(f"合并PDF时出错: {str(e)}")
                    continue
            
            # 生成输出文件名
            if is_archive:
                # 如果是压缩包，使用压缩包名称（不含扩展名）作为输出文件名
                base_name = Path(input_path).stem
                if output_path:
                    output_file = os.path.join(output_path, f"{base_name}.pdf")
                else:
                    # 使用压缩包所在目录作为输出目录
                    output_file = os.path.join(Path(input_path).parent, f"{base_name}.pdf")
            else:
                # 普通目录处理
                if output_path:
                    output_file = os.path.join(output_path, f"{os.path.basename(input_path)}.pdf")
                else:
                    output_file = os.path.join(input_path, f"{os.path.basename(input_path)}.pdf")
            
            # 保存最终的PDF
            merger.write(output_file)
            merger.close()
            
            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"处理完成: {output_file}")
            logger.info(f"处理用时: {processing_time:.2f}秒")
            
            return output_file
            
        except Exception as e:
            logger.error(f"处理过程中出错: {str(e)}")
            return None
            
        finally:
            # 清理临时文件
            if not use_memory:
                for temp_pdf in temp_pdfs:
                    try:
                        if temp_pdf and os.path.exists(temp_pdf):
                            os.remove(temp_pdf)
                    except Exception as e:
                        logger.error(f"清理临时PDF文件时出错: {str(e)}")
                try:
                    if temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.error(f"清理临时目录时出错: {str(e)}")
            
            # 清理内存文件创建的临时文件
            if memory_files:
                for file in image_files:
                    try:
                        if os.path.exists(file):
                            os.remove(file)
                    except Exception as e:
                        logger.error(f"清理内存文件创建的临时文件时出错: {str(e)}")
            
            # 清理资源
            cleanup_resources()
    
    except Exception as e:
        logger.error(f"处理过程中出现未预期的错误: {str(e)}")
        return None

def is_archive_file(file_path):
    """检查文件是否为支持的压缩包格式"""
    archive_extensions = {'.zip', '.rar', '.7z'}
    return Path(file_path).suffix.lower() in archive_extensions

def extract_archive(archive_path, extract_dir=None, use_memory=False):
    """解压压缩包到指定目录或内存中"""
    archive_path = Path(archive_path)
    suffix = archive_path.suffix.lower()
    
    try:
        # 添加 RAR 环境检查
        if suffix == '.rar':
            if not rarfile.UNRAR_TOOL:
                raise RuntimeError(
                    "需要安装 WinRAR 并添加至系统 PATH\n"
                    "下载地址：https://www.rarlab.com/download.htm\n"
                    "安装后请将 C:\\Program Files\\WinRAR 添加到系统环境变量"
                )
        if use_memory:
            # 使用内存解压
            if suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    # 创建内存文件系统
                    memory_fs = {}
                    for file_info in zip_ref.infolist():
                        if not file_info.is_dir():
                            memory_fs[file_info.filename] = zip_ref.read(file_info.filename)
                    return memory_fs
            elif suffix == '.7z':
                with py7zr.SevenZipFile(archive_path, 'r') as sz:
                    # 创建内存文件系统
                    memory_fs = {}
                    for file_info in sz.getnames():
                        if not file_info.endswith('/'):
                            memory_fs[file_info] = sz.read([file_info])[file_info]
                    return memory_fs
            elif suffix == '.rar':
                with rarfile.RarFile(archive_path, 'r') as rar_ref:
                    # 处理中文文件名编码
                    rar_ref.encoding = 'gbk'  # Windows 专用编码
                    
                    # 创建内存文件系统
                    memory_fs = {}
                    for file_info in rar_ref.infolist():
                        if not file_info.isdir():
                            try:
                                # 双重解码处理中文路径
                                filename = file_info.filename.encode('cp437').decode('gbk')
                                content = rar_ref.read(file_info)
                                memory_fs[filename] = content
                            except Exception as e:
                                # 备选方案直接使用原始文件名
                                filename = file_info.filename
                                memory_fs[filename] = content
                    return memory_fs
            else:
                raise ValueError(f"不支持的压缩包格式: {suffix}")
        else:
            # 使用临时目录解压
            if not extract_dir:
                # 在压缩包所在目录创建temp文件夹
                extract_dir = archive_path.parent / 'temp'
                if not extract_dir.exists():
                    extract_dir.mkdir(parents=True)
            
            if suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif suffix == '.7z':
                with py7zr.SevenZipFile(archive_path, 'r') as sz:
                    sz.extractall(extract_dir)
            elif suffix == '.rar':
                with rarfile.RarFile(archive_path, 'r') as rar_ref:
                    rar_ref.extractall(extract_dir)
            else:
                raise ValueError(f"不支持的压缩包格式: {suffix}")
            return str(extract_dir)
    except Exception as e:
        print(f"\n解压文件失败: {archive_path}")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        return None

def process_input_path(input_path, output_path, target_size, device, ppi, keep_original_ratio=True, keep_original_size=False, use_memory=False, sort_by='name', show_file_list=False, logger=None):
    """处理输入路径（可能是目录或压缩包）"""
    input_path = Path(input_path)
    
    # 如果是压缩包
    if is_archive_file(input_path):
        print(f"\n检测到压缩包: {input_path}")
        
        if use_memory:
            # 使用内存解压
            print("正在解压到内存中...")
            memory_fs = extract_archive(input_path, use_memory=True)
            if not memory_fs:
                return None
            
            # 处理内存中的文件
            return process_directory(
                str(input_path),
                output_path,
                target_size,
                device,
                ppi,
                keep_original_ratio,
                keep_original_size,
                use_memory,
                sort_by,
                show_file_list,
                memory_fs,
                True,  # 标记为压缩包处理
                logger
            )
        else:
            # 使用临时目录解压
            temp_dir = extract_archive(input_path)
            if not temp_dir:
                return None
            
            try:
                # 处理解压后的目录
                result = process_directory(
                    temp_dir,
                    output_path,
                    target_size,
                    device,
                    ppi,
                    keep_original_ratio,
                    keep_original_size,
                    use_memory,
                    sort_by,
                    show_file_list,
                    None,
                    True,  # 标记为压缩包处理
                    logger
                )
                
                # 删除临时目录
                try:
                    shutil.rmtree(temp_dir)
                    print(f"已删除临时目录: {temp_dir}")
                except Exception as e:
                    print(f"删除临时目录失败: {str(e)}")
                
                return result
            except Exception as e:
                # 确保在出错时也删除临时目录
                try:
                    shutil.rmtree(temp_dir)
                    print(f"已删除临时目录: {temp_dir}")
                except:
                    pass
                raise e
    # 如果是目录
    elif input_path.is_dir():
        return process_directory(
            str(input_path),
            output_path,
            target_size,
            device,
            ppi,
            keep_original_ratio,
            keep_original_size,
            use_memory,
            sort_by,
            show_file_list,
            None,
            False,  # 标记为普通目录处理
            logger
        )
    else:
        print(f"\n无效的输入路径: {input_path}")
        return None

def main():
    # 记录开始时间
    start_time = time.time()
    # 设置日志记录器
    logger = setup_logger()
    logger.info("=== 开始图片转PDF处理 ===")
    
    parser = argparse.ArgumentParser(description='将图片转换为PDF文件')
    parser.add_argument('input_paths', type=str, nargs='+',
                      help='可输入多个路径，支持目录和压缩包(.zip/.rar/.7z)')
    parser.add_argument('--output', '-o', type=str, 
                      help='输出目录路径（可选）')
    parser.add_argument('--size', '-s', type=str, default='A4',
                      choices=list(PAPER_SIZES.keys()),
                      help='预设纸张尺寸，默认A4')
    parser.add_argument('--orientation', type=str, default='portrait',
                      choices=['portrait', 'landscape'],
                      help='纸张方向：portrait（竖版）, landscape（横版），默认portrait')
    parser.add_argument('--ppi', type=int, default=250,
                      help='图片分辨率（每英寸像素数），默认250')
    parser.add_argument('--custom-size', action='store_true',
                      help='启用自定义尺寸模式')
    parser.add_argument('--width', type=int,
                      help='自定义PDF页面宽度（像素），需要与--custom-size一起使用')
    parser.add_argument('--height', type=int,
                      help='自定义PDF页面高度（像素），需要与--custom-size一起使用')
    parser.add_argument('--no-keep-original-ratio',action='store_false',
                      help='不保持原图比例，将图片拉伸至指定尺寸')
    parser.add_argument('--keep-original-size', action='store_true',
                      help='保持原图大小，忽略所有尺寸设置')
    parser.add_argument('--use-memory', action='store_true',
                      help='使用内存存储临时PDF文件（默认使用磁盘存储）')
    parser.add_argument('--no-gpu', action='store_true',
                      help='强制使用CPU处理（默认使用GPU如果可用）')
    parser.add_argument('--sort-by', type=str, default='name',
                      choices=['name', 'created', 'modified', 'size'],
                      help='文件排序方式：name(文件名), created(创建时间), modified(修改时间), size(文件大小)，默认name')
    parser.add_argument('--show-files', action='store_true',
                      help='显示排序后的文件列表')
    parser.add_argument('--delete-source', action='store_true',
                      help='转换成功后删除源文件/文件夹')
    
    args = parser.parse_args()
    
    # 如果没有提供输入路径，显示使用说明
    if not args.input_paths:
        print_usage()
        return
    
    # 验证自定义尺寸参数
    if args.custom_size:
        if not args.width or not args.height:
            logger.error("错误：使用--custom-size时必须同时指定--width和--height")
            return
        if args.width <= 0 or args.height <= 0:
            logger.error("错误：宽度和高度必须大于0")
            return
    elif args.width or args.height:
        logger.error("错误：使用--width或--height时必须同时使用--custom-size")
        return
    
    
    # 获取设备
    device = torch.device('cpu') if args.no_gpu else get_device()
    logger.info(f"使用设备: {device}")
    
    
    # 确定目标尺寸
    if args.keep_original_size:
        # 保持原图大小，使用默认尺寸（实际会被忽略）
        target_size = (2100, 2970)  # A4尺寸的像素值
        logger.info("注意：由于启用了--keep-original-size，将保持原图大小")
    elif args.custom_size:
        # 使用自定义尺寸
        target_size = (args.width, args.height)
        logger.info(f"使用自定义尺寸: {args.width}x{args.height} 像素")
    else:
        # 使用预设尺寸
        width_mm, height_mm = PAPER_SIZES[args.size]
        # 根据方向调整尺寸
        if args.orientation == 'landscape':
            width_mm, height_mm = height_mm, width_mm
        target_size = (
            mm_to_pixels(width_mm, args.ppi),
            mm_to_pixels(height_mm, args.ppi)
        )
        logger.info(f"使用预设尺寸: {args.size} ({args.orientation})")
        logger.info(f"纸张尺寸: {width_mm}x{height_mm} 毫米")
    
    logger.info(f"目标分辨率: {args.ppi} PPI")
    logger.info(f"输出尺寸: {target_size[0]}x{target_size[1]} 像素")
    logger.info(f"保持原图比例: {'是' if args.no_keep_original_ratio else '否'}")
    logger.info(f"保持原图大小: {'是' if args.keep_original_size else '否'}")
    logger.info(f"使用内存存储: {'是' if args.use_memory else '否'}")
    logger.info(f"排序方式: {args.sort_by}")
    logger.info(f"转换后删除源文件: {'是' if args.delete_source else '否'}")
    
    for input_path in args.input_paths:
        input_path = input_path.strip()
        if not os.path.exists(input_path):
            logger.error(f"路径不存在: {input_path}")
            continue
        
        logger.info(f"\n处理输入: {input_path}")
        output_file = process_input_path(
            input_path, 
            args.output, 
            target_size, 
            device, 
            args.ppi, 
            args.no_keep_original_ratio,
            args.keep_original_size,
            args.use_memory,
            args.sort_by,
            args.show_files,
            logger
        )
        
        if output_file:
            logger.info(f"PDF已生成: {output_file}")
            # 如果启用了删除源文件选项且PDF生成成功
            if args.delete_source:
                try:
                    if os.path.isfile(input_path):
                        os.remove(input_path)
                        logger.info(f"已删除源文件: {input_path}")
                    elif os.path.isdir(input_path):
                        shutil.rmtree(input_path)
                        logger.info(f"已删除源文件夹: {input_path}")
                except Exception as e:
                    logger.error(f"删除源文件/文件夹时出错: {str(e)}")
        
        # 清理资源
        cleanup_resources()
    
    # 计算并显示总运行时间
    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    
    logger.info("\n程序运行时间统计:")
    if hours > 0:
        logger.info(f"总运行时间: {hours}小时 {minutes}分钟 {seconds:.2f}秒")
    elif minutes > 0:
        logger.info(f"总运行时间: {minutes}分钟 {seconds:.2f}秒")
    else:
        logger.info(f"总运行时间: {seconds:.2f}秒")
    
    logger.info("=== 处理完成 ===")

if __name__ == '__main__':
    main()