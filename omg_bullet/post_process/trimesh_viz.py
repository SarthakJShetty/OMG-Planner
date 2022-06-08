import argparse
import trimesh
import pickle
import imageio
import numpy as np
import io
import os
from PIL import Image

def pkls_to_gif(pkls_dir):
    images = []
    for file_name in sorted(os.listdir(f'{pkls_dir}/pred_pkls'), key=lambda x: int(x.replace('pred_', '').replace('.pkl', ''))):
        if file_name.endswith('.pkl'):
            print(file_name)
            scene_dict = pickle.load(open(f'{pkls_dir}/pred_pkls/{file_name}', 'rb'))
            scene = trimesh.exchange.load.load_kwargs(scene_dict)
            data = scene.save_image(resolution=(320,240))
            image = np.array(Image.open(io.BytesIO(data))) 
            images.append(image)
    for i in range(15):
        images.append(image)
    imageio.mimsave(f"{pkls_dir}/pred.gif", images)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="Base directory of pred_pkls/ folder to process pkl trimesh files into gif", type=str)
    args = parser.parse_args()

    pkls_to_gif(args.dir)
