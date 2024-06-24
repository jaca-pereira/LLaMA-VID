import math
import os

import json
import traceback
from dataclasses import field, dataclass
from random import random

import spacy
import torch
import transformers
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from llamavid.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX, \
    IMAGE_TOKEN_INDEX
from llamavid.model.multimodal_encoder.eva_vit import EVAVisionTowerLavis
from typing import Dict, Optional, Sequence
from torch.utils.data import Dataset
import pickle

from llamavid.model.to_me.text_token_pruning import text_topk_pruning
from llamavid.model.to_me.token_selection import bipartite_soft_matching
from llamavid.train.train import DataArguments, preprocess, make_supervised_data_module, ModelArguments, \
    TrainingArguments
from llamavid import conversation as conversation_lib

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 data_args: DataArguments, image_processor: torch.nn.Module):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.image_processor = image_processor

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        attempt, max_attempt = 0, 10
        image = None
        while attempt < max_attempt:
            try:
                sources = self.list_data_dict[i]
                if isinstance(i, int):
                    sources = [sources]
                assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
                if 'image' in sources[0]:
                    image_file = self.list_data_dict[i]['image']
                    image_folder = self.data_args.image_folder
                    img_path = os.path.join(image_folder, image_file)
                    # convert image type for OCR VQA dataset
                    if image_file is not None:
                        if 'ocr' in image_file:
                            if not os.path.exists(img_path):
                                image_file = image_file.replace(".jpg", ".png")
                                img_path = os.path.join(image_folder, image_file)

                    image = Image.open(img_path).convert('RGB')
                    if self.data_args.image_aspect_ratio == 'pad':
                        def expand2square(pil_img, background_color):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result

                        image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
                        image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    else:
                        image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

                elif 'video' in sources[0]:
                    video_file = self.list_data_dict[i]['video']
                    video_folder = self.data_args.video_folder
                    video_file = os.path.join(video_folder, video_file)
                    if not os.path.exists(video_file):
                        # Load the video file
                        video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.pkl']
                        for fmt in video_formats:  # Added this line
                            temp_path = f"{video_file.split('.')[0]}{fmt}"
                            if os.path.exists(temp_path):
                                video_file = temp_path
                                img_path = video_file
                                print("Loading video file: ", video_file)
                                break
                        else:
                            print('File {} not exist!'.format(video_file.split('.')[0]))

                    vr = VideoReader(video_file, ctx=cpu(0))
                    sample_fps = round(vr.get_avg_fps() / self.data_args.video_fps)
                    frame_idx = [i for i in range(0, len(vr), sample_fps)]
                    video = vr.get_batch(frame_idx).asnumpy()
                    image = self.image_processor.preprocess(video, return_tensors='pt')['pixel_values']
                break
            except Exception as e:
                traceback.print_exc()
                attempt += 1
                print(f"Error in loading {i}, retrying...")
                i = random.randint(0, len(self.list_data_dict) - 1)

        has_image = ('image' in self.list_data_dict[i]) or ('video' in self.list_data_dict[i])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=has_image,
            prompt=self.data_args.input_prompt,
            refine_prompt=self.data_args.refine_prompt)
        if 'prompt' in data_dict:
            prompt = data_dict['prompt']
        else:
            prompt = None

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['image_file'] = img_path
        elif 'video' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['image_file'] = video_file
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['image_file'] = None

        # prompt exist in the data
        if prompt is not None:
            data_dict['prompt'] = prompt


        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images) and len(images) > 1:
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        if 'prompt' in instances[0]:
            batch['prompts'] = [instance['prompt'] for instance in instances]

        return batch


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    feat_dir: str = field(default=None, metadata={"help": "Path to save the extracted features."})
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    input_prompt: Optional[str] = field(default=None)


class TokenSelection:
    def __init__(self, max_length=4096, mm_redundant_token_selection="sum"):
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.text_model.eval()
        self.text_model.requires_grad_(False)
        self.text_model.to('cuda')
        self.text_token_pruning = text_topk_pruning
        self.text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.nlp = spacy.load("en_core_web_sm")
        self.mm_redundant_token_selection = mm_redundant_token_selection
        self.max_length = max_length
        def extract_nouns_adjectives_verbs(text):
            # Process the text
            doc = self.nlp(text)

            # Initialize lists to store nouns, adjectives, and verbs
            nouns_adjectives_verbs = []

            # Iterate through tokens in the processed document
            for token in doc:
                # Check if the token is a noun
                if token.pos_ == 'NOUN':
                    for child in token.children:
                        if child.pos_ == 'ADJ':
                            nouns_adjectives_verbs.append(f'{child.text} {token.text}')
                # Check if the token is a verb
                elif token.pos_ == 'VERB':
                    nouns_adjectives_verbs.append(token.text)

                return nouns_adjectives_verbs

        self.extract_nouns_adjectives_verbs = extract_nouns_adjectives_verbs

    def merge_and_prune_tokens(self, image_feature, prompt, input_ids):

        image_feature = image_feature[..., 1:, :]
        num_frames = image_feature.shape[0]
        if num_frames == 1:
            kth = 1
        else:
            kth = 2
        image_feature = image_feature.reshape(image_feature.shape[0] * image_feature.shape[1], image_feature.shape[2])
        for ki in range(kth):
            image_feature = [image_feature[i: i + (16*256)] if i+(16*256) < len(image_feature) else image_feature[i:] for i in range(0, len(image_feature), (16*256))]
            image_feature = [clip.unsqueeze(0) for clip in image_feature]
            merges = [bipartite_soft_matching(metric=clip, r=clip.shape[1]//2) for clip in image_feature]
        new_image_feature = [merge[0](clip, mode=self.mm_redundant_token_selection)[0] for
                             merge, clip in zip(merges, image_feature)]
        image_feature = torch.cat(new_image_feature, dim=0)
        text_inputs = self.extract_nouns_adjectives_verbs(prompt)
        text_inputs = self.text_tokenizer(text_inputs, return_tensors="pt", truncation=True, padding=True).to(device=image_feature.device)
        text_embeds = self.text_model(**text_inputs).last_hidden_state
        text_embeds = text_embeds.type(image_feature.dtype)
        num_non_image_tokens = (input_ids != IMAGE_TOKEN_INDEX).sum()
        topk = self.max_length - num_non_image_tokens - 1
        if topk > image_feature.shape[0]:
            topk = image_feature.shape[0]
        image_feature = self.text_token_pruning(image_feature, text_embeds, topk)

        return image_feature


def initialize_dataset_and_tokenizer():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank


    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    orig_rope_scaling = getattr(config, "rope_scaling", None)
    if orig_rope_scaling is None:
        orig_rope_scaling = {"factor": 1}

    orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len:
        orig_ctx_len *= orig_rope_scaling_factor
        if training_args.model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}


    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens(dict(pad_token="[PAD]"))
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)

    if model_args.mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
                                              special_tokens=True)

    # Initialize the CLIP model
    vision_tower = EVAVisionTowerLavis(model_args.vision_tower, model_args.image_processor, args=None).cuda()
    vision_tower.eval()

    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    data_args.image_processor = vision_tower.image_processor

    # initialize token merging HERE
    token_selection = TokenSelection(training_args.model_max_length, model_args.mm_redundant_token_selection)
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)

    return data_module, token_selection, data_args.feat_dir



def main():
    data_module, token_selection, feat_dir = initialize_dataset_and_tokenizer()
    os.makedirs(feat_dir, exist_ok=True)

    # Extract features and perform token merging and pruning
    for batch in tqdm(data_module):
        image = batch['images']
        selected_tokens = token_selection.merge_and_prune_tokens(image)
        # Save the results
        feat_path = os.path.join(feat_dir, f'{os.path.basename(video_path)}.pkl')
        with open(feat_path, 'wb') as f:
            pickle.dump(selected_tokens, f)

if __name__ == "__main__":
    main()