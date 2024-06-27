#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2023 Yanwei Li
# ------------------------------------------------------------------------
import math
import pickle
import time
from abc import ABC, abstractmethod
import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llamavid.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN

from transformers import CLIPTextModel, CLIPTokenizer

from .to_me.text_token_pruning import text_topk_pruning
from .to_me.token_selection import merge_source, merge_wavg, bipartite_soft_matching
import spacy


class LLaMAVIDMetaModel:

    def __init__(self, config):
        super(LLaMAVIDMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None, max_token=2048):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower
        self.config.image_processor = getattr(model_args, 'image_processor', None)

        vision_tower = build_vision_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.max_token = max_token
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))



    def initialize_token_selection(self):
        self.config.mm_text_token_pruning = getattr(self.config, 'mm_text_token_pruning', True)  # TODO change to false once config is set
        if self.config.mm_text_token_pruning :
            self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
            self.text_model.eval()
            self.text_model.requires_grad_(False)
            self.text_model = self.text_model.to(self.device)
            self.text_token_pruning = text_topk_pruning
            self.text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            self.nlp = spacy.load("en_core_web_sm")

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
                                nouns_adjectives_verbs.append(f'A photo of a {child.text} {token.text}')
                        nouns_adjectives_verbs.append(f'A photo of a {token.text}')
                    # Check if the token is a verb
                    elif token.pos_ == 'VERB':
                        nouns_adjectives_verbs.append(f'A photo of a {token.text}')

                return nouns_adjectives_verbs
            self.extract_nouns_adjectives_verbs = extract_nouns_adjectives_verbs
        self.config.mm_token_source = getattr(self.config, 'mm_token_source', False) #True if source tokens are to be plotted
        self.config.mm_redundant_token_selection = getattr(self.config, 'mm_redundant_token_selection', "sum")  #Options: "mean", "amax", "sum". "None"

class LLaMAVIDMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def merge_tokens(self, image_features):
        num_frames = image_features.shape[0]
        base_size = num_frames * 32
        bucket_size = 2 ** math.floor(math.log2(base_size)) #nearest power of 2
        reduction_ratio = max(0, math.floor(math.log2(bucket_size / 512)))
        image_features = image_features.reshape(image_features.shape[0] * image_features.shape[1], image_features.shape[2])
        for kt in range(reduction_ratio):
            merge = bipartite_soft_matching(metric=image_features, bucket_size=bucket_size)
            image_features, _ = merge_wavg(merge, image_features)
            if kt < reduction_ratio - 1:
                base_size = image_features.shape[0]//256 * 32
                bucket_size = 2 ** math.floor(math.log2(base_size))
        return image_features

    def prune_tokens(self, image_features, prompt, input_ids):
        normalized_image_features = F.normalize(image_features)
        text_inputs = self.get_model().text_tokenizer(self.get_model().extract_nouns_adjectives_verbs(prompt[0]), return_tensors="pt", truncation=True, padding=True).to(image_features.device)
        text_embeds = self.get_model().text_model(**text_inputs).last_hidden_state
        text_embeds = text_embeds.type(image_features.dtype)
        topk = self.config.max_length - (input_ids != IMAGE_TOKEN_INDEX).sum()
        if topk > image_features.shape[0]:
            topk = image_features.shape[0]
        text_embeds = F.normalize(text_embeds)
        return self.get_model().text_token_pruning(image_features, normalized_image_features, text_embeds, topk)

    def merge_and_prune_tokens(self, image_feature, prompts, input_ids, images=None):

        num_frames = image_feature.shape[0]
        base_size = num_frames * 32
        bucket_size = 2 ** math.floor(math.log2(base_size))
        if bucket_size <= 512:
            reduction_ratio = 1
        elif bucket_size <= 2048:
            reduction_ratio = 2
        elif bucket_size <= 4096:
            reduction_ratio = 3
        else:
            reduction_ratio = 4
        image_feature = image_feature.reshape(image_feature.shape[0] * image_feature.shape[1], image_feature.shape[2])
        for kt in range(reduction_ratio):
            merge, _ = bipartite_soft_matching(metric=image_feature, r=image_feature.shape[0]//reduction_ratio, bucket_size=bucket_size)
            image_feature, _ = merge_wavg(merge, image_feature)

        if self.config.mm_token_source:
            sources = merge_source(merge, image_feature)[0]

        text_inputs = None
        for prompt in prompts:
            if text_inputs is None:
                text_inputs = self.get_model().text_tokenizer(self.get_model().extract_nouns_adjectives_verbs(prompt), return_tensors="pt", truncation=True, padding=True).to(image_feature.device)
            else:
                text_inputs_i = self.get_model().text_tokenizer(self.get_model().extract_nouns_adjectives_verbs(prompt), return_tensors="pt", truncation=True, padding=True).to(image_feature.device)
                text_inputs = torch.cat([text_inputs, text_inputs_i], dim=0)
        text_embeds = self.get_model().text_model(**text_inputs).last_hidden_state
        text_embeds = text_embeds.type(image_feature.dtype)

        if self.config.mm_token_source:
            topk = 50
            self.get_model().text_token_pruning(image_feature, text_embeds, topk, images, sources)

        num_non_image_tokens = (input_ids != IMAGE_TOKEN_INDEX).sum()

        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            topk = self.config.max_length - num_non_image_tokens - 1
        else:
            topk = self.config.max_length - num_non_image_tokens

        if topk > image_feature.shape[0]:
            topk = image_feature.shape[0]
        image_feature = self.get_model().text_token_pruning(image_feature, text_embeds, topk)
        return image_feature

    def update_prompt(self, prompts=None):
        self.prompts = prompts

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, attention_mask, past_key_values, labels, images, prompts=None, indexes = None
    ):
        if prompts is None and hasattr(self, 'prompts'):
            prompts = self.prompts

        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                            dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            images = [image if len(image.shape) == 4 else image.unsqueeze(0) for image in images]
            image_counts = [image.shape[0] for image in images]
            concat_images = torch.cat(images, dim=0)
            image_features = self.get_model().get_vision_tower()(concat_images)
        else:
            images = [image.unsqueeze(0) for image in images]
            image_counts = [image.shape[0] for image in images]
            concat_images = torch.cat(images, dim=0)
            image_features = self.get_model().get_vision_tower()(concat_images)

        if self.config.mm_vision_select_feature == 'patch':
            if image_features.shape[-2] % 2 == 1:
                image_features = image_features[..., 1:, :]

        if image_counts is not None:
            image_features = list(image_features.split(image_counts))
        num_inps = len(image_features) #batch size
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2

                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])

                if cur_image_idx + 1 < num_inps:
                    cur_image_idx += 1

                continue

            cur_image_features = image_features[cur_image_idx]
            cur_image_features = self.merge_tokens(cur_image_features)
            cur_image_features = self.prune_tokens(cur_image_features, prompts[batch_idx], cur_input_ids)
            with open(indexes[batch_idx], 'wb') as f:
                pickle.dump(cur_image_features, f)
            cur_image_features = self.get_model().mm_projector(cur_image_features)

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            while image_token_indices.numel() > 0:

                image_token_start = image_token_indices[0]


                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config,
                                                                                  'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start - 1]).detach())
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[image_token_start - 1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[image_token_start + 1:image_token_start + 2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                       dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start + 1])
                        cur_labels = cur_labels[image_token_start + 2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                       dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start + 1:]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config,
                                                                                  'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start + 2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            # changle image idx after processing one sample
            if cur_image_idx + 1 < num_inps:
                cur_image_idx += 1
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config,
                                                                                  'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)

            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)


        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed,
                                           torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label,
                                               torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX,
                                                          dtype=cur_new_label.dtype, device=cur_new_label.device)),
                                              dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            # only used for right padding in tokenlizer
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels,
                                                                                    new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True,
                                                        dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                                                         False, dtype=attention_mask.dtype,
                                                         device=attention_mask.device)
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            # only used for right padding in tokenlizer
            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
