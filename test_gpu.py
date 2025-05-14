import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import os

def test_gpu_acceleration():
    """测试GPU加速功能"""
    print("\n=== GPU加速测试 ===")
    
    # 检查CUDA是否可用
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前CUDA版本: {torch.version.cuda}")
        print(f"当前GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"当前可用显存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    # 创建测试图片
    print("\n创建测试图片...")
    test_image = Image.new('RGB', (2000, 2000), color='white')
    test_image.save('test_image.png')
    
    # 定义图像处理转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((1000, 1000), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
    ])
    
    # 测试CPU处理
    print("\n测试CPU处理...")
    start_time = time.time()
    for _ in range(10):
        img_tensor = transform(test_image)
        # 模拟一些处理
        img_tensor = img_tensor * 1.5
        img_tensor = torch.clamp(img_tensor, 0, 1)
    cpu_time = time.time() - start_time
    print(f"CPU处理时间: {cpu_time:.2f}秒")
    
    # 测试GPU处理
    if torch.cuda.is_available():
        print("\n测试GPU处理...")
        device = torch.device('cuda')
        start_time = time.time()
        for _ in range(10):
            img_tensor = transform(test_image).to(device)
            # 模拟一些处理
            img_tensor = img_tensor * 1.5
            img_tensor = torch.clamp(img_tensor, 0, 1)
            # 确保GPU操作完成
            torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        print(f"GPU处理时间: {gpu_time:.2f}秒")
        print(f"加速比: {cpu_time/gpu_time:.2f}倍")
    
    # 清理测试文件
    try:
        os.remove('test_image.png')
    except:
        pass
    
    print("\n=== 测试完成 ===")

if __name__ == '__main__':
    test_gpu_acceleration() 