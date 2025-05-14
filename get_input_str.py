# import sys
# import pyperclip

# def collect_inputs():
#     """收集所有输入来源：命令行参数 + 标准输入流"""
#     inputs = sys.argv[1:]  # 初始化为命令行参数
    
#     # 检测是否存在管道输入或需要交互式输入
#     if not sys.stdin.isatty():
#         # 处理管道输入（自动读取所有行）
#         inputs += [line.strip().strip('"') for line in sys.stdin if line.strip()]
#     else:
#         # 交互式输入模式（仅在需要时提示）
#         if not inputs:
#             print("请输入文件路径（每行一个，Ctrl+Z回车结束）：")
#         else:
#             print("可以继续输入更多路径（Ctrl+Z回车结束）：")
        
#         # 持续读取直到EOF
#         while True:
#             try:
#                 line = input()
#                 if line.strip():
#                     inputs.append(line.strip().strip('"'))
#             except (EOFError, KeyboardInterrupt):
#                 break
    
#     return inputs

# def format_paths(path_list):
#     """为每个路径添加系统级引号"""
#     return ' '.join(f'"{path}"' for path in path_list)

# def main():
#     try:
#         # 收集所有输入路径
#         all_paths = collect_inputs()
        
#         if not all_paths:
#             raise ValueError("没有检测到任何输入路径")
        
#         # 生成最终命令字符串
#         combined = format_paths(all_paths)
        
#         # 剪贴板操作
#         pyperclip.copy(combined)
#         print("\n生成结果（已复制到剪贴板）：")
#         print(combined)
        
#         return 0
#     except Exception as e:
#         print(f"错误发生：{e}", file=sys.stderr)
#         return 1

# if __name__ == "__main__":
#     sys.exit(main())
import pyperclip

# 输入的多行字符串（注意要使用原始字符串避免转义问题）
text = r'''

'''

# 处理过程
lines = [line.strip() for line in text.splitlines() if line.strip()]  # 分割并清理空行
combined = " ".join(lines)  # 用空格连接

try:
    pyperclip.copy(combined)
    print("以下内容已成功复制到剪贴板：\n")
    print(combined)
except ImportError:
    print("错误：需要先安装 pyperclip 库")
    print("请执行：pip install pyperclip")
    print("\n生成的字符串：\n" + combined)
except Exception as e:
    print(f"复制时发生错误：{str(e)}")
    print("\n生成的字符串：\n" + combined)