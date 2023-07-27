import os
import cv2
import numpy as np
import argparse

# EXAMPLE:
# python3 compute_PSNR_for_evaluation_set.py /home/racoon/Desktop/test_compute_PSNR/folder_1 /home/racoon/Desktop/test_compute_PSNR/folder_2

def calculate_psnr(img1, img2):
    # Compute PSNR
    return cv2.PSNR(img1, img2)

def main(folder1, folder2):
    psnr_values = []

    files1 = sorted(os.listdir(folder1))
    files2 = sorted(os.listdir(folder2))

    # Ensure both directories contain the same number of images
    assert len(files1) == len(files2), "Directories must contain the same number of images"

    for file1, file2 in zip(files1, files2):
        img1 = cv2.imread(os.path.join(folder1, file1))
        img2 = cv2.imread(os.path.join(folder2, file2))
        
        if img1 is None or img2 is None:
            print(f"Could not read one of the images: {file1} or {file2}")
            continue

        # Make sure both images have the same size
        if img1.shape != img2.shape:
            print(f"Images {file1} and {file2} do not have the same size, skipping...")
            continue

        psnr_values.append(calculate_psnr(img1, img2))

    avg_psnr = np.mean(psnr_values)
    print(f"The average PSNR of the images in {folder1} and {folder2} is: {avg_psnr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate average PSNR between images in two folders')
    parser.add_argument('folder1', type=str, help='Path to first folder')
    parser.add_argument('folder2', type=str, help='Path to second folder')
    
    args = parser.parse_args()
    
    main(args.folder1, args.folder2)
