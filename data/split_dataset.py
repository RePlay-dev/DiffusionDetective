import argparse
import os
import random
import shutil
from pathlib import Path


def split_dataset(src_dir, train_dir, valid_dir, test_dir, valid_ratio=0.15, test_ratio=0.15, seed=42, copy=True):
    """
    Split the dataset into train, validation, and test sets, maintaining arbitrary subfolder structure.

    :param src_dir: Source directory containing all images
    :param train_dir: Directory to store training images
    :param valid_dir: Directory to store validation images
    :param test_dir: Directory to store test images
    :param valid_ratio: Ratio of validation set (default: 0.15)
    :param test_ratio: Ratio of test set (default: 0.15)
    :param seed: Random seed for reproducibility
    :param copy: If True, copy files instead of moving them
    """
    random.seed(seed)

    # Create train, validation, and test directories if they don't exist
    for d in [train_dir, valid_dir, test_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Get all subdirectories (classes) in the source directory
    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

    def process_directory(dir_path, class_name, relative_path=""):
        image_files = []
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path):
                process_directory(item_path, class_name, os.path.join(relative_path, item))
            elif item.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append((item_path, relative_path))

        if image_files:
            random.shuffle(image_files)

            # Calculate split indices
            n_files = len(image_files)
            n_test = int(n_files * test_ratio)
            n_valid = int(n_files * valid_ratio)

            # Split files
            test_files = image_files[:n_test]
            valid_files = image_files[n_test:n_test + n_valid]
            train_files = image_files[n_test + n_valid:]

            # Function to copy or move files
            def transfer_files(files, dest_dir):
                for src, rel_path in files:
                    dest_folder = str(os.path.join(dest_dir, class_name, rel_path))
                    Path(dest_folder).mkdir(parents=True, exist_ok=True)
                    dest = os.path.join(dest_folder, os.path.basename(src))
                    shutil.copy2(src, dest) if copy else shutil.move(src, dest)

            # Transfer files to respective directories
            transfer_files(train_files, train_dir)
            transfer_files(valid_files, valid_dir)
            transfer_files(test_files, test_dir)

            print(
                f"Class {class_name}, Subfolder {relative_path}: {len(train_files)} training, {len(valid_files)} validation, {len(test_files)} test images.")

    for class_name in classes:
        class_dir = os.path.join(src_dir, class_name)
        process_directory(class_dir, class_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, validation, and test sets")
    parser.add_argument("src_dir", help="Source directory containing all images")
    parser.add_argument("--train_dir", default="training/train", help="Directory to store training images")
    parser.add_argument("--valid_dir", default="training/valid", help="Directory to store validation images")
    parser.add_argument("--test_dir", default="test", help="Directory to store test images")
    parser.add_argument("--valid_ratio", type=float, default=0.15, help="Ratio of validation set")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Ratio of test set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")

    args = parser.parse_args()

    split_dataset(args.src_dir, args.train_dir, args.valid_dir, args.test_dir,
                  args.valid_ratio, args.test_ratio, args.seed, not args.move)
