import argparse
import math
import os
import spacy
import torch
import transformers
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from llamavid.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, \
    IMAGE_TOKEN_INDEX
from llamavid.model.multimodal_encoder.eva_vit import EVAVisionTowerLavis
import pickle
from llamavid.model.to_me.text_token_pruning import text_topk_pruning
from llamavid.model.to_me.token_selection import bipartite_soft_matching, merge_wavg
from llamavid.train.train import make_supervised_data_module
from llamavid import conversation as conversation_lib

class TokenSelection:
    def __init__(self, max_length=4096, mm_redundant_token_selection="sum"):
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.text_model.eval()
        self.text_model.requires_grad_(False)
        self.text_model.to('cuda:0')
        self.text_token_pruning = text_topk_pruning
        self.text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.nlp = spacy.load("en_core_web_sm")
        self.mm_redundant_token_selection = mm_redundant_token_selection
        self.max_length = max_length
        def extract_nouns_adjectives_verbs(text, nlp):
            # Process the text
            doc = nlp(text)

            # Initialize lists to store nouns, adjectives, and verbs
            nouns_adjectives_verbs = []

            # Iterate through tokens in the processed document
            for token in doc:
                # Check if the token is a noun
                if token.pos_ == 'NOUN':
                    for child in token.children:
                        if child.pos_ == 'ADJ':
                            nouns_adjectives_verbs.append(f'A photo of a {child.text} {token.text}')
                    nouns_adjectives_verbs.append(f'A photo of a {token.text}')
                # Check if the token is a verb
                elif token.pos_ == 'VERB':
                    nouns_adjectives_verbs.append(f'A photo of a {token.text}')

            return nouns_adjectives_verbs

        self.extract_nouns_adjectives_verbs = extract_nouns_adjectives_verbs

    def merge_and_prune_tokens(self, image_feature, prompts, input_ids):

        image_feature = image_feature[..., 1:, :]
        num_frames = image_feature.shape[0]
        if num_frames == 1:
            kth = 1
        else:
            kth = 2
        image_feature = image_feature.reshape(image_feature.shape[0] * image_feature.shape[1], image_feature.shape[2])
        for ki in range(kth):
            merge, _ = bipartite_soft_matching(metric=image_feature, r=image_feature.shape[0]//2)
            image_feature, _ = merge_wavg(merge, image_feature)
        text_embeds = None
        for prompt in prompts:
            text_inputs = self.extract_nouns_adjectives_verbs(prompt, self.nlp)
            text_inputs = self.text_tokenizer(text_inputs, return_tensors="pt", truncation=True, padding=True).to(device=image_feature.device)
            if text_embeds is None:
                text_embeds = self.text_model(**text_inputs).last_hidden_state
                text_embeds = text_embeds.type(image_feature.dtype)
            else:
                text_embeds_i = self.text_model(**text_inputs).last_hidden_state
                text_embeds_i = text_embeds_i.type(image_feature.dtype)
                text_embeds = torch.cat([text_embeds, text_embeds_i], dim=0)

        num_non_image_tokens = (input_ids != IMAGE_TOKEN_INDEX).sum()
        topk = self.max_length - num_non_image_tokens - 1
        if topk > image_feature.shape[0]:
            topk = image_feature.shape[0]
        image_feature = self.text_token_pruning(image_feature, text_embeds, topk)
        return image_feature

def parse_args():
    parser = argparse.ArgumentParser(description="Extract features for training.")
    parser.add_argument("--model_name_or_path", type=str, default="model_zoo/LLM/vicuna/13B-V1.5")
    parser.add_argument("--text_model_name_or_path", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--model_max_length", type=int, default=8192)
    parser.add_argument("--version", type=str, default="imgsp_v1")
    parser.add_argument("--mm_use_im_patch_token", type=bool, default=False)
    parser.add_argument("--mm_use_im_start_end", type=bool, default=False)
    parser.add_argument("--vision_tower", type=str, default="./model_zoo/LAVIS/eva_vit_g.pth")
    parser.add_argument("--image_processor", type=str, default="./llamavid/processor/clip-patch14-224")
    parser.add_argument("--image_aspect_ratio", type=str, default="square")
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--feat_dir", type=str, default="./data/LLaMA-VID-Finetune/clip-text-token-reduction-full-features")
    parser.add_argument("--video_fps", type=int, default=1)
    parser.add_argument("--video_folder", type=str, default="./data/LLaMA-VID-Finetune")
    parser.add_argument("--image_folder", type=str, default="./data/LLaMA-VID-Finetune")
    parser.add_argument("--data_path", type=str, default="./data/LLaMA-VID-Finetune/llava_v1_5_mix665k_with_video_chatgpt_maxtime_5min.json")
    parser.add_argument("--mm_redundant_token_selection", type=str, default="sum")
    parser.add_argument("--cache_dir", type=str, default="~/.cache")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--refine_prompt", type=bool, default=False)
    parser.add_argument("--input_prompt", type=str, default=None)
    parser.add_argument("--is_multimodal", type=bool, default=True)
    return parser.parse_args()


def initialize_dataset_and_tokenizer():

    args = parse_args()
    config = transformers.AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    orig_rope_scaling = getattr(config, "rope_scaling", None)
    if orig_rope_scaling is None:
        orig_rope_scaling = {"factor": 1}

    orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len:
        orig_ctx_len *= orig_rope_scaling_factor
        if args.model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(args.model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}


    if 'mpt' in args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            model_max_length=args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if args.version == "v0":
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens(dict(pad_token="[PAD]"))
    elif args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if args.mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)

    if args.mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
                                              special_tokens=True)

    # Initialize the CLIP model
    vision_tower = EVAVisionTowerLavis(args.vision_tower, args.image_processor, args=None).cuda()
    vision_tower.eval()
    vision_tower.to(dtype=torch.bfloat16 if args.bf16 else torch.float16, device=args.device)
    args.image_processor = vision_tower.image_processor

    # initialize token merging HERE
    token_selection = TokenSelection(args.model_max_length, args.mm_redundant_token_selection)
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=args)

    return data_module, token_selection, vision_tower, args.feat_dir



def main():
    data_module, token_selection, vision_tower, feat_dir = initialize_dataset_and_tokenizer()
    os.makedirs(feat_dir, exist_ok=True)
    total_steps = len(data_module['train_dataset'])
    print_every = 10
    progress_bar = tqdm(total=total_steps, desc='Extracting features and performing token merging and pruning')
    for i, instance in enumerate(data_module['train_dataset']):
        batch = data_module['data_collator'].__call__([instance])
        images = batch['images'][0]
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        image_feature = vision_tower(images)
        selected_tokens = token_selection.merge_and_prune_tokens(image_feature, batch['prompts'][0], batch['input_ids'][0])
        # Save the results
        feat_path = os.path.join(feat_dir, f"{instance['index']}.pkl")
        with open(feat_path, 'wb') as f:
            pickle.dump(selected_tokens, f)
        del selected_tokens
        torch.cuda.empty_cache()
        if i % print_every == 0:
            progress_bar.set_postfix({'index': i})
            progress_bar.update()
if __name__ == "__main__":
    main()
