# 图片转PDF工具

一个功能强大的命令行工具，用于将图片转换为PDF文件。支持多种图片格式，可以保持原图比例或自定义尺寸，支持GPU加速处理。

## 功能特点

- 支持多种图片格式（JPG、JPEG、PNG、BMP、GIF）
- 支持多个输入目录和压缩包（使用空格分隔）
- 支持多种预设纸张尺寸（A4、A5、A3、B5、B4、Letter、Legal）
- 支持自定义PDF页面尺寸
- 支持自定义PPI（每英寸像素数）
- 支持保持原图比例或拉伸至指定尺寸
- 支持保持原图大小（忽略所有尺寸设置）
- 支持内存存储临时文件（提高处理速度）
- 支持GPU加速处理（如果可用）
- 支持多种文件排序方式（文件名、创建时间、修改时间、文件大小）
- 支持显示排序后的文件列表
- 支持自然文件名排序

## 安装

1. 确保已安装Python 3.6或更高版本
2. 克隆或下载此仓库
3. 安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python image_to_pdf.py "输入路径"
```

### 完整参数说明

```bash
python image_to_pdf.py "输入路径" [选项]
```

#### 参数说明

- `输入路径`：可以是单个路径或多个路径（用空格分隔），可以是文件夹也可以是压缩包（输入压缩包出现问题尝试解压后输入文件夹）
  - 例如：`F:\图片文件夹` 或 `"F:\图片1" "F:\图片2"`

#### 选项

- `--output`, `-o`：输出目录路径（可选）
- `--size`, `-s`：预设纸张尺寸，可选值：
  - A4, A5, A3, B5, B4, Letter, Legal
  - 默认：A4
- `--ppi`：图片分辨率（每英寸像素数）
  - 默认：250
- `--custom-size`：启用自定义尺寸模式
  - 需要同时指定 `--width`和 `--height`
- `--width`：自定义PDF页面宽度（像素）
  - 需要与 `--custom-size`一起使用
- `--height`：自定义PDF页面高度（像素）
  - 需要与 `--custom-size`一起使用
- `--keep-original-ratio`：保持原图比例，将图片拉伸至指定尺寸
  - 默认不保持原图比例
- `--keep-original-size`：保持原图大小，忽略所有尺寸设置
- `--use-memory`：使用内存存储临时PDF文件
  - 默认使用磁盘存储
- `--no-gpu`：强制使用CPU处理
  - 默认使用GPU（如果可用）
- `--sort-by`：文件排序方式，可选值：
  - name（文件名）
  - created（创建时间）
  - modified（修改时间）
  - size（文件大小）
  - 默认：name
- `--show-files`：显示排序后的文件列表
- `--help`, `-h`：显示帮助信息

### 使用示例

1. 基本用法（不保持原图比例，按文件名排序）：

```bash
python image_to_pdf.py "F:\图片文件夹"
```

2. 使用自定义尺寸：

```bash
python image_to_pdf.py "F:\图片文件夹" --custom-size --width 1920 --height 1080
```

3. 保持原图比例：

```bash
python image_to_pdf.py "F:\图片文件夹" --keep-original-ratio
```

4. 保持原图大小：

```bash
python image_to_pdf.py "F:\图片文件夹" --keep-original-size
```

5. 使用内存存储：

```bash
python image_to_pdf.py "F:\图片文件夹" --use-memory
```

6. 按创建时间排序并显示文件列表：

```bash
python image_to_pdf.py "F:\图片文件夹" --sort-by created --show-files
```

7. 使用A5纸张，300PPI：

```bash
python image_to_pdf.py "F:\图片文件夹" --size A5 --ppi 300
```

8. 处理多个目录：

```bash
python image_to_pdf.py "F:\图片1;F:\图片2" --output "F:\输出目录"
```

### 注意事项

1. 默认情况下不保持原图比例，将图片缩放至指定尺寸
2. 使用 `--keep-original-ratio`可以保持原图比例，将图片拉伸至指定尺寸
3. 使用 `--keep-original-size`可以保持原图大小，忽略所有尺寸设置
4. 使用 `--use-memory`可以将临时PDF文件存储在内存中，提高处理速度
5. 程序会自动检测是否可以使用GPU，如果可用则使用GPU加速
6. 文件会按照指定的方式排序（默认按文件名）
7. 使用 `--show-files`选项可以查看排序后的文件列表

## 依赖

- Pillow>=10.0.0
- PyPDF2>=3.0.0
- torch>=2.0.0
- torchvision>=0.15.0
- tqdm>=4.65.0
- numpy>=1.22.0,<1.28.0
