import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore.nn import TrainOneStepCell, WithLossCell
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import os
import time
import csv
from tqdm import tqdm
from model import UNet  # 从 model.py 导入 UNet
from dataLoader import create_dataset  # 从 dataLoader.py 导入 create_dataset

Mean = [0.4970002770423889, 0.5053070783615112, 0.4676517844200134]
Std = [0.24092243611812592, 0.23609396815299988, 0.25256040692329407]

def denormalize(tensor, mean, std):
    tensor = tensor.copy()
    for t, m, s in zip(tensor, mean, std):
        t *= s
        t += m
    return tensor

def rgb_to_ycbcr(img):
    img = img.astype(np.float32)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    y = 65.738 * r / 256 + 129.057 * g / 256 + 25.064 * b / 256 + 16
    cb = -37.945 * r / 256 - 74.494 * g / 256 + 112.439 * b / 256 + 128
    cr = 112.439 * r / 256 - 94.154 * g / 256 - 18.285 * b / 256 + 128

    ycbcr_img = np.stack([y, cb, cr], axis=-1)
    ycbcr_img = np.clip(ycbcr_img, 0, 255).astype(np.uint8)
    return ycbcr_img

def calculate_metrics(output, target):
    output = output.asnumpy().transpose(1, 2, 0)
    target = target.asnumpy().transpose(1, 2, 0)
    
    output = (output * 255.0).clip(0, 255).astype(np.uint8)
    target = (target * 255.0).clip(0, 255).astype(np.uint8)
    
    output_ycbcr = rgb_to_ycbcr(output)
    target_ycbcr = rgb_to_ycbcr(target)
    
    output_y = output_ycbcr[:, :, 0]
    target_y = target_ycbcr[:, :, 0]
    
    psnr_value = psnr(target_y, output_y, data_range=255)
    ssim_value = ssim(target_y, output_y, data_range=255)
    
    return psnr_value, ssim_value

class L1CharbonnierLoss(nn.Cell):
    def __init__(self):
        super(L1CharbonnierLoss, self).__init__()
        self.eps = 1e-3

    def construct(self, X, Y):
        diff = X - Y
        error = ops.sqrt(diff * diff + self.eps)
        loss = ops.mean(error)
        return loss

data_dir = 'archive'
best_model_path = 'unet_best.ckpt'
batch_size = 32
num_epochs = 1000
learning_rate = 1e-4
weight_decay = 1e-5
log_file = f'training_log_{time.strftime("%Y%m%d_%H%M%S")}.csv'
MAX_LOSS_THRESHOLD = 10.0
LOSS_CHECK_PATIENCE = 50

train_dataset = create_dataset(data_dir, 'train', batch_size, shuffle=True)
test_dataset = create_dataset(data_dir, 'test', 1, shuffle=False)

model = UNet(in_channels=3, out_channels=3)
if os.path.exists(best_model_path):
    mindspore.load_checkpoint(best_model_path, model)

optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate, weight_decay=weight_decay)
criterion = L1CharbonnierLoss()

net_with_loss = WithLossCell(model, criterion)
train_network = TrainOneStepCell(net_with_loss, optimizer)

with open(log_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Test PSNR", "Test SSIM", "Learning Rate"])

def train_step(inputs, targets):
    loss = train_network(inputs, targets)
    return loss

def train_epoch(model, train_dataset, recent_losses):
    model.set_train(True)
    running_loss = 0.0
    num_batches = train_dataset.get_dataset_size()
    
    iterator = train_dataset.create_dict_iterator()
    for batch in tqdm(iterator, total=num_batches):
        rain_images = batch["rain"].astype(mindspore.float32)
        clean_images = batch["clean"].astype(mindspore.float32)
        
        loss = train_step(rain_images, clean_images)
        
        if loss.asnumpy() > MAX_LOSS_THRESHOLD:
            print(f"\nLoss explosion detected: {loss.asnumpy():.4f}")
            return None, True
        
        running_loss += loss.asnumpy()
        
        recent_losses.append(loss.asnumpy())
        if len(recent_losses) > LOSS_CHECK_PATIENCE:
            recent_losses.pop(0)
        
        if len(recent_losses) == LOSS_CHECK_PATIENCE:
            if np.mean(recent_losses[-10:]) > np.mean(recent_losses[:10]) * 2:
                print("\nContinuous loss increase detected")
                return None, True
    
    return running_loss / num_batches, False

def test_model(model, test_dataset):
    model.set_train(False)
    total_psnr, total_ssim = 0.0, 0.0
    num_images = test_dataset.get_dataset_size()
    
    iterator = test_dataset.create_dict_iterator()
    for batch in tqdm(iterator, total=num_images, desc='Testing'):
        rain_images = batch["rain"].astype(mindspore.float32)
        clean_images = batch["clean"].astype(mindspore.float32)
        
        output = model(rain_images)
        denorm_output = denormalize(output[0], mean=Mean, std=Std)
        denorm_clean = denormalize(clean_images[0], mean=Mean, std=Std)
        
        psnr_value, ssim_value = calculate_metrics(denorm_output, denorm_clean)
        total_psnr += psnr_value
        total_ssim += ssim_value
    
    return total_psnr / num_images, total_ssim / num_images

best_psnr = -1
recent_losses = []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    avg_train_loss, loss_exploded = train_epoch(model, train_dataset, recent_losses)
    
    if loss_exploded:
        print("Restarting training from last best model...")
        if os.path.exists(best_model_path):
            mindspore.load_checkpoint(best_model_path, model)
            optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate, weight_decay=weight_decay)
            net_with_loss = WithLossCell(model, criterion)
            train_network = TrainOneStepCell(net_with_loss, optimizer)
            recent_losses = []
            continue
        else:
            print("No best model found to reload. Stopping training.")
            break
    
    avg_psnr, avg_ssim = test_model(model, test_dataset)
    current_lr = learning_rate
    
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, avg_train_loss, avg_psnr, avg_ssim, current_lr])
    
    print(f"Loss: {avg_train_loss:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LR: {current_lr:.6f}")
    
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        mindspore.save_checkpoint(model, best_model_path)
        print(f"New best PSNR: {best_psnr:.4f}, model saved.")