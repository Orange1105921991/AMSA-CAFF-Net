import json
from os.path import join
import glob


if __name__ == '__main__':
    # path to folder that contains images
    img_folder = './datasets/overall/test/images'

    # path to the final json file
    output_json = './jsons/val.json'

    img_list = []

    for img_path in glob.glob(join(img_folder,'*.png')):
        img_list.append(img_path)

    with open(output_json,'w') as f:
        json.dump(img_list,f)
