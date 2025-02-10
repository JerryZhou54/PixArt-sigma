import os
from PIL import Image, ImageDraw, ImageFont

def stitch_images(input_folders, output_folder, image_ids, prompts):
    assert len(input_folders) == len(image_ids)
    assert len(image_ids[0]) == len(prompts)
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取文件夹中图片的数量和文件名（假设所有文件夹中图片数量和名称都相同）
    # image_names = sorted([f for f in os.listdir(input_folders[0]) if f.endswith('.jpg')])

    for i in range(len(prompts)):
        images_to_stitch = []
        for j in range(len(input_folders)):
            img_dir = f'{image_ids[j][i]}.jpg'
            image = Image.open(os.path.join(input_folders[j],img_dir))
            images_to_stitch.append(image)
        
        # 计算拼接后图像的宽高
        total_width = sum(img.width for img in images_to_stitch)
        max_height = max(img.height for img in images_to_stitch)
        title_height = 0
        
        # 创建新的空白图像用于拼接
        stitched_image = Image.new('RGB', (total_width, max_height + title_height))

        draw = ImageDraw.Draw(stitched_image)
        font = ImageFont.truetype("arial.ttf", size=20)
        bbox = draw.textbbox((0, 0), prompts[i], font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]        
        text_x = (total_width - text_width) // 2  # Center the text
        text_y = (title_height - text_height) // 2  # Center vertically within the title area
        draw.text((text_x, text_y), prompts[i], fill=(255, 255, 255), font=font)
        
        # 将图片逐个粘贴到空白图像中
        x_offset = 0
        for img in images_to_stitch:
            stitched_image.paste(img, (x_offset, title_height))
            x_offset += img.width
        
        # 保存拼接后的图像
        stitched_image.save(os.path.join(output_folder, f'{image_ids[0][i]}.jpg'))
        print(f"Saved stitched image to {output_folder}")

# 使用示例
input_folders = ["/home/hyou/Efficient-Diffusion/PixArt-sigma/output/results/step-2500/dpm-solver/cfg4_steps20_seed0/pred",
"/home/hyou/Efficient-Diffusion/PixArt-sigma/output/results/step-8500/dpm-solver/cfg4_steps20_seed0/pred",
"/home/hyou/Efficient-Diffusion/PixArt-sigma/output/results/diffmod-timewise/dpm-solver/cfg4_steps20_seed0/pred"]  # 输入的文件夹路径列表，按顺序排列
output_folder = "/home/hyou/Efficient-Diffusion/PixArt-sigma/output/visual/"                    # 输出文件夹路径
image_ids = [[10110,10075,10072,10070,10029,10019,10018,9980,9947,9931,9925,9920,9914,9831], [52,54,59,95,135,138,154,319], [10110,10075,10071,10070,10029,10019,10017,9981,9947,9930,9925,9921,9914,9831]]
image_ids = [[10093, 9989, 9970, 9926, 9888, 9835, 9779, 9771, 9740, 9722, 9712, 9709, 9704, 9698, 9679, 9557, 9536, 9533, 9504, 9498, 9479, 9409, 9367, 9339, 9300, 9299, 9274, 9273, 9250, 9220, 9198, 9148, 9142, 9125],
[10094, 9988, 9969, 9924, 9889, 9835, 9778, 9772, 9739, 9722, 9711, 9708, 9704, 9698, 9678, 9556, 9535, 9536, 9505, 9498, 9478, 9409, 9367, 9342, 9300, 9299, 9275, 9273, 9250, 9220, 9196, 9149, 9140, 9124]]
prompts = ["alternative christmas wedding styled shoot in reds with large feather headpiece", 
"a stylish maternity session with a delicate lace and blue tulle gown",
"strawberry rhubarb cake with almonds and pomegranate",
"bmw e36 wallpaper full hd",
"herb weller and pat explore the sculpted crevasses of robson glacier",
"this abandoned asylum is still standing and still creepy",
"afternoon winter glow",
"a little owl took advantage of a downpour to spread its wings and bathe"]

stitch_images(input_folders, output_folder, image_ids, prompts)