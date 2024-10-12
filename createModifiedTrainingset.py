import os
import random
from PIL import Image, ImageDraw, ImageEnhance

def adjust_brightness(image_path, factor, output_path=""):
    base, ext = os.path.splitext(output_path)

    with Image.open(image_path) as img:
        enhancer = ImageEnhance.Brightness(img)
        img_enhanced = enhancer.enhance(factor)

        dir = os.path.dirname(output_path)
        if dir and not os.path.exists(dir):
            os.makedirs(dir)


        output_path = f"{base}_brightness{ext}"

        img_enhanced.save(output_path)

def occlude_image(image_path, output_path=""):
    img = Image.open(image_path)
   
    # Choose a random position for the top-left corner of the 8x8 square
    max_position = 64 - 12
    x = random.randint(0, max_position)
    y = random.randint(0, max_position)
    
    draw = ImageDraw.Draw(img)
    
    # Draw the 8x8 black square
    draw.rectangle([x, y, x+12, y+12], fill="black")

    base, ext = os.path.splitext(output_path)

    dir = os.path.dirname(output_path)
    if dir and not os.path.exists(dir):
        os.makedirs(dir)

    output_path = f"{base}_occluded{ext}"

    img.save(output_path)


def process_images(input_dir, output):
    for dirname in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, dirname)
        output_dir = output
        output_dir = os.path.join(output_dir, dirname)
        for filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, filename)
            dir = os.path.join(output_dir, filename)

            adjustment_options = [0.5, 0.3, 0.2]
            adjustment_factor = random.choice(adjustment_options)

            factor  = 1 + random.choice([-1, 1]) * adjustment_factor


            d = os.path.dirname(dir)
            if d and not os.path.exists(d):
                os.makedirs(d)

            # save the original
            img = Image.open(img_path)
            img.save(dir)


            # List of the functions with their parameters
            functions = [
                lambda: occlude_image(image_path=img_path, output_path=dir),
                lambda: adjust_brightness(image_path=img_path, output_path=dir, factor=factor)
            ]

            random.choice(functions)()


# change path to where you set the dataset
input_directory = '/home/gras/Downloads/archive/64'
output_directory = '/home/gras/Downloads/archive/modifiedDataset'

process_images(input_directory, output_directory)