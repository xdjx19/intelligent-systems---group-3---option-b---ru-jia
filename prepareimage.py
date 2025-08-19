from PIL import Image
import numpy as np
import sys
import csv
import os

def prepareimage(image_path, threshold=128):
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Can't open image: {image_path}: {e}")
        return None
    
    img_gray = img.convert('L')
    img_resized = img_gray.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized)
    bimatrix = (img_array >= threshold).astype(int)
    
    invmatrix = 1 - bimatrix
    return invmatrix

def exportcsv(matrix, output_file):
    try:
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(matrix)
        print(f"Output exported to {output_file}")
    except Exception as e:
        print(f"Error exporting: {e}")

def processimage(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            bimatrix = prepareimage(image_path)
            
            if bimatrix is not None:
                output_csv_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.csv")
                exportcsv(bimatrix, output_csv_path)
            else:
                print(f"Can't process image {filename}.")

def main():
    if len(sys.argv) != 3:
        print("Usage: python prepareimage.py <input_folder> <output_folder>")
        return
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    processimage(input_folder, output_folder)

if __name__ == "__main__":
    main()
