import pickle
import torch

class VideoProcessor:
    def __init__(self, video_file, default_image_token, video_token):
        self.video_file = video_file
        self.default_image_token = default_image_token
        self.video_token = video_token

    def process_video(self):
        video_info = pickle.load(open(self.video_file, 'rb'))
        image = torch.from_numpy(video_info['feats'][:, 1:])
        input_prompt = video_info['inputs'].replace('...', '')
        # replace the default image token with multiple tokens
        input_prompt = input_prompt.replace(self.default_image_token,
                                            self.default_image_token * self.video_token)
        return image, input_prompt

class Main:
    def __init__(self, video_file, default_image_token, video_token):
        self.processor = VideoProcessor(video_file, default_image_token, video_token)

    def run(self):
        image, input_prompt = self.processor.process_video()
        # You can now use image and input_prompt as needed
        print(image, input_prompt)

if __name__ == "__main__":
    main = Main('~/data/datasets/ucf_crime/Videos/Abuse/Abuse014_x264.avi', 'DEFAULT_IMAGE_TOKEN', 'video_token')
    main.run()