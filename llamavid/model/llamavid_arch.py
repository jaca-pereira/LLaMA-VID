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
from .to_me.token_selection import kth_bipartite_soft_matching, merge_source, merge_wavg, bipartite_soft_matching
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
            self.text_model.to(self.vision_tower.device)
            self.text_token_pruning = text_topk_pruning
            self.text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            self.nlp = spacy.load("en_core_web_sm")

            def extract_nouns_adjectives_verbs(nlp, text):
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
                                nouns_adjectives_verbs.append(f'{child.text} {token.text}')
                    # Check if the token is a verb
                    elif token.pos_ == 'VERB':
                        nouns_adjectives_verbs.append(token.text)

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

    def encode_images(self, images, prompts=None, long_video=False, input_ids = None):
        if long_video:
            # use pre-computed features
            image_features = images
        else:
            image_features = self.get_model().get_vision_tower()(images)

        # remove cls embedding
        if self.config.mm_vision_select_feature == 'patch':
            if image_features.shape[-2] % 2 == 1:
                image_features = image_features[..., 1:, :]

        image_features_list = []
        if image_features.ndim == 3:
            image_features = image_features.unsqueeze(0)
        for i, image_feature in enumerate(image_features):
            if self.config.mm_redundant_token_selection is not None:
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
                    if self.config.mm_token_source:
                        sources = [merge_source(merge, clip)[0] for merge, clip in zip(merges, image_feature)]
                    new_image_feature = [merge[0](clip, mode=self.config.mm_redundant_token_selection)[0] for
                                         merge, clip in zip(merges, image_feature)]
                    image_feature = torch.cat(new_image_feature, dim=0)
                if self.config.mm_token_source:
                    if len(sources) > 1 and sources[-1].shape[-1] != sources[-2].shape[-1]:
                        sources.pop(-1)
                        new_image_feature.pop(-1)
                        image_feature = torch.cat(new_image_feature, dim=0)
                    sources = torch.cat(sources, dim=0)
                del new_image_feature
            if self.config.mm_text_token_pruning:
                text_inputs = self.get_model().extract_nouns_adjectives_verbs(self.get_model().nlp, prompts[i][0])
                text_inputs = self.get_model().text_tokenizer(text_inputs, return_tensors="pt", truncation=True, padding=True).to(device=self.device)
                text_embeds = self.get_model().text_model(**text_inputs).last_hidden_state
                text_embeds = text_embeds.half()
                num_non_image_tokens = (input_ids[i] != IMAGE_TOKEN_INDEX).sum()
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    topk = self.config.max_length - num_non_image_tokens - 1
                else:
                    topk = self.config.max_length - num_non_image_tokens

                if topk > image_feature.shape[0]:
                    topk = image_feature.shape[0]

                if self.config.mm_token_source:
                    topk = 50
                    if images.ndim == 4:
                        image_feature = self.get_model().text_token_pruning(image_feature, text_embeds, topk, images, sources)
                    else:
                        image_feature = self.get_model().text_token_pruning(image_feature, text_embeds, topk, images[i], sources)
                else:
                    image_feature = self.get_model().text_token_pruning(image_feature, text_embeds, topk)

            image_feature = self.get_model().mm_projector(image_feature)
            image_features_list.append(image_feature)

        return image_features_list

    def update_prompt(self, prompts=None):
        self.prompts = prompts

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, attention_mask, past_key_values, labels, images, prompts=None
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

        # pre-process images for long video
        if images[0].shape[-1] > 1000:
            long_video = True
        else:
            long_video = False

        if type(images) is list or images.ndim == 5:
            # not reseshape for long video
            if not long_video:
                images = [image if len(image.shape) == 4 else image.unsqueeze(0) for image in images]
            concat_images = torch.cat(images, dim=0)
            image_features = self.encode_images(concat_images, prompts, long_video=long_video, input_ids = input_ids)
        else:
            image_features = self.encode_images(images, prompts, long_video=long_video, input_ids = input_ids)

        num_inps = len(image_features)
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                if isinstance(image_features, list):
                    cur_image_features = image_features[cur_image_idx][0]
                else:
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

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            if not long_video:
                while image_token_indices.numel() > 0:
                    if isinstance(image_features, list):
                        cur_image_features = image_features[cur_image_idx]
                    else:
                        cur_image_features = image_features
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
            else:
                cur_new_input_embeds = torch.Tensor(len(cur_input_ids), self.config.hidden_size).to(dtype=self.dtype,
                                                                                                    device=self.device)
                text_token_indices = torch.where(cur_input_ids != IMAGE_TOKEN_INDEX)[0]
                if not self.training and self.get_model().embed_tokens.weight.device != cur_input_ids.device:
                    model_device = self.get_model().embed_tokens.weight.device
                    data_device = cur_input_ids.device
                    cur_input_ids_text = cur_input_ids[text_token_indices].to(device=model_device)
                    cur_new_input_embeds[text_token_indices] = self.get_model().embed_tokens(cur_input_ids_text).to(
                        device=data_device)
                else:
                    cur_new_input_embeds[text_token_indices] = self.get_model().embed_tokens(
                        cur_input_ids[text_token_indices])
                cur_image_features = image_features[cur_image_idx]
                cur_new_input_embeds[image_token_indices] = cur_image_features
                new_input_embeds.append(cur_new_input_embeds)
                if labels is not None:
                    new_labels.append(cur_labels)
                if cur_image_idx + 1 < num_inps:
                    cur_image_idx += 1

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
