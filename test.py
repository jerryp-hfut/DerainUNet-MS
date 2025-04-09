import os
import numpy as np
from PIL import Image
import mindspore
from mindspore import context
from mindspore import load_checkpoint, load_param_into_net
from model import UNet  # 确保model.py在同一目录下

# # 设置MindSpore上下文
# context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# DeRaindrop数据集的均值和标准差
Mean = [0.4970002770423889, 0.5053070783615112, 0.4676517844200134]
Std = [0.24092243611812592, 0.23609396815299988, 0.25256040692329407]

def normalize(image):
    """将图像归一化处理"""
    image = image.copy().astype(np.float32) / 255.0
    image = (image - Mean) / Std
    return image

def denormalize(image):
    """将归一化的图像转换回原始像素值范围"""
    image = image.copy()
    for i in range(3):
        image[i] = image[i] * Std[i] + Mean[i]
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image

def process_image(model, image_path, output_dir):
    """处理单张图像"""
    # 读取图像
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    
    # 记录原始尺寸
    original_height, original_width = img_np.shape[:2]
    
    # 归一化并调整为模型输入格式
    img_np = normalize(img_np)
    img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
    
    # 创建MindSpore张量并添加批次维度
    input_tensor = mindspore.Tensor(np.expand_dims(img_np, axis=0), mindspore.float32)
    
    # 模型推理
    output_tensor = model(input_tensor)
    
    # 获取结果并转换回图像格式
    output_np = output_tensor[0].asnumpy()  # 移除批次维度
    output_np = denormalize(output_np)  # 反归一化
    output_np = output_np.transpose(1, 2, 0)  # CHW -> HWC
    
    # 如果需要，将输出调整为原始尺寸
    if output_np.shape[:2] != (original_height, original_width):
        output_img = Image.fromarray(output_np)
        output_img = output_img.resize((original_width, original_height), Image.LANCZOS)
        output_np = np.array(output_img)
    
    # 保存结果
    output_img = Image.fromarray(output_np)
    
    # 获取原始文件名（不含路径和扩展名）
    base_name = os.path.basename(image_path)
    file_name, ext = os.path.splitext(base_name)
    
    # 确定输出路径
    output_path = os.path.join(output_dir, f"{file_name}_derained{ext}")
    output_img.save(output_path)
    
    print(f"处理完成: {base_name} -> {os.path.basename(output_path)}")
    
    return output_path

def batch_process(model_path, input_dir, output_dir, supported_extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """批量处理图像"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    model = UNet(in_channels=3, out_channels=3)
    param_dict = load_checkpoint(model_path)
    load_param_into_net(model, param_dict)
    model.set_train(False)  # 设置为评估模式
    
    # 获取输入目录中所有图像
    image_paths = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_extensions):
            image_paths.append(os.path.join(input_dir, filename))
    
    if not image_paths:
        print(f"在 {input_dir} 中未找到支持的图像文件")
        return
    
    print(f"找到 {len(image_paths)} 张图像，开始处理...")
    
    # 处理每张图像
    processed_images = []
    for image_path in image_paths:
        try:
            output_path = process_image(model, image_path, output_dir)
            processed_images.append((image_path, output_path))
        except Exception as e:
            print(f"处理 {image_path} 时出错: {str(e)}")
    
    print(f"所有处理完成！共处理 {len(processed_images)} 张图像，输出目录: {output_dir}")
    
    return processed_images

if __name__ == "__main__":
    # 设置路径
    model_path = "unet_best.ckpt"  # 模型权重路径
    input_dir = "imgs"            # 输入图像文件夹
    output_dir = "results"        # 输出结果文件夹
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件 {model_path} 不存在!")
    else:
        # 批量处理图像
        batch_process(model_path, input_dir, output_dir)