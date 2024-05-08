import json
from tqdm import tqdm


def write_json(json_data, output_file):
    json.dump(json_data, open(output_file, 'w'), indent=2)


if __name__ == '__main__':
    llava_file = 'data/LLaMA-VID-Finetune/llava_v1_5_mix665k_with_video_chatgpt_maxtime_5min.json'
    llava_file_features = 'data/LLaMA-VID-Finetune/llava_v1_5_mix665k_with_video_chatgpt_maxtime_5min_features.json'
    # llava_file = 'images/llava_v1_5_mix665k_with_video_chatgpt.json'
    llava_data = json.load(open(llava_file, 'r'))

    for item in tqdm(llava_data):
        if 'video' in item.keys():
            item['video'] = item['video'].replace('mp4', 'pkl')

    write_json(llava_data, llava_file_features)
