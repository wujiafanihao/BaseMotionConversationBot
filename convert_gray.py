import os
import cv2
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

train_path = "archive/data_relabeled_balanced_1x/train"

def convert_to_gray_and_save(image_path, output_path):
    """
    读取 image_path 指定的图片，将其转换为灰度图后保存到 output_path。
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片：{image_path}")
        return
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 保存灰度图像
    cv2.imwrite(output_path, gray)

def process_directory(train_dir):
    """
    遍历 train 目录下的所有子目录，对每个目录中的图片进行灰度转换，
    并将处理后的图片存储到同目录下添加 _gray 后缀的子目录中。
    使用多线程加快处理速度，并显示进度条。
    """
    # 筛选出所有子目录
    categories = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    for cat in categories:
        src_folder = os.path.join(train_dir, cat)
        dst_folder = os.path.join(train_dir, cat + "_gray")
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        
        # 选择常见格式的图片
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(src_folder, ext)))
        
        total_files = len(image_files)
        if total_files == 0:
            print(f"目录 {src_folder} 内未找到有效图片。")
            continue
        
        print(f"开始处理 {cat} 文件夹，共 {total_files} 张图片...")
        futures = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            for image_file in image_files:
                file_name = os.path.basename(image_file)
                dst_file = os.path.join(dst_folder, file_name)
                futures.append(executor.submit(convert_to_gray_and_save, image_file, dst_file))
            
            # 使用 tqdm 显示进度条
            for _ in tqdm(as_completed(futures), total=total_files, desc=f"Processing {cat}"):
                pass
        
        print(f"目录 {src_folder} 处理完毕，灰度图片保存在 {dst_folder}")

    print("所有图片灰度处理完成！")

if __name__ == "__main__":
    # 指定 train 目录，确保目录路径正确
    process_directory(train_path)