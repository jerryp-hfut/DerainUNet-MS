import os
from PIL import Image
import numpy as np
import random
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

Mean = [0.4970002770423889, 0.5053070783615112, 0.4676517844200134]
Std = [0.24092243611812592, 0.23609396815299988, 0.25256040692329407]

class DerainDataset:
    def __init__(self, data_dir, split, crop_size=(256, 256)):
        self.data_dir = data_dir
        self.split = split
        self.crop_size = crop_size
        self.image_pairs = self._make_dataset()

    def _make_dataset(self):
        image_pairs = []
        if self.split == 'train':
            data_subdir = 'data_train'
            gt_subdir = 'gt_train'
        elif self.split == 'test':
            data_subdir = 'data_test'
            gt_subdir = 'gt_test'
        else:
            raise ValueError("Invalid split name, expected 'train' or 'test'")

        data_dir_path = os.path.join(self.data_dir, 'data', data_subdir)
        gt_dir_path = os.path.join(self.data_dir, 'gt', gt_subdir)

        for filename in os.listdir(data_dir_path):
            if filename.endswith('_rain.png'):
                gt_filename = filename.split('_rain.png')[0] + '_clean.png'
                rain_image_path = os.path.join(data_dir_path, filename)
                clean_image_path = os.path.join(gt_dir_path, gt_filename)

                if os.path.exists(clean_image_path):
                    image_pairs.append((rain_image_path, clean_image_path))
                else:
                    print(f"Warning: GT file not found for {filename}, skipping this pair.")

        if not image_pairs:
            raise RuntimeError(f"No valid image pairs found in {data_dir_path} and {gt_dir_path}")

        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        rain_image_path, clean_image_path = self.image_pairs[idx]
        rain_image = Image.open(rain_image_path).convert('RGB')
        clean_image = Image.open(clean_image_path).convert('RGB')

        rain_image = np.array(rain_image).astype(np.float32) / 255.0
        clean_image = np.array(clean_image).astype(np.float32) / 255.0

        if self.split == 'train':
            rain_image = vision.Resize((240, 360))(rain_image)
            clean_image = vision.Resize((240, 360))(clean_image)

            if random.random() > 0.5:
                rain_image = np.fliplr(rain_image)
                clean_image = np.fliplr(clean_image)
            if random.random() > 0.5:
                rain_image = np.flipud(rain_image)
                clean_image = np.flipud(clean_image)

        rain_image = (rain_image - Mean) / Std
        clean_image = (clean_image - Mean) / Std

        rain_image = rain_image.transpose(2, 0, 1)
        clean_image = clean_image.transpose(2, 0, 1)

        return rain_image, clean_image

def create_dataset(data_dir, split, batch_size, shuffle=True):
    dataset = DerainDataset(data_dir=data_dir, split=split, crop_size=(256, 256))
    ms_dataset = ds.GeneratorDataset(
        dataset,
        column_names=["rain", "clean"],
        shuffle=shuffle
    )
    ms_dataset = ms_dataset.batch(batch_size, drop_remainder=True if split == 'train' else False)
    return ms_dataset

if __name__ == "__main__":
    data_dir = 'archive'
    train_dataset = create_dataset(data_dir, 'train', 32, shuffle=True)
    for data in train_dataset.create_dict_iterator():
        rain_images, clean_images = data["rain"], data["clean"]
        print(f"Rain image shape: {rain_images.shape}, Clean image shape: {clean_images.shape}")
        break