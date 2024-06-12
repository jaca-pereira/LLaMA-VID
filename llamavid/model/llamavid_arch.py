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
import random
from abc import ABC, abstractmethod

import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llamavid.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from .to_me.token_merging import kth_bipartite_soft_matching, merge_wavg, merge_source
from .to_me.token_pruning import text_topk_pruning, plot_source_top_k_tokens



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

        self.config.mm_redundant_token_selection = getattr(self.config, 'mm_redundant_token_selection', "Pruning") #Options: "Merging", "AvgPooling", "MaxPooling", "None"
        self.config.mm_token_source = getattr(self.config, 'mm_token_source', False) #True if source tokens are to be plotted
        if self.config.mm_redundant_token_selection == "Merging":
            self.token_selection = kth_bipartite_soft_matching
            self.token_selection_method = merge_wavg
            self.token_source = merge_source if self.config.mm_token_source else None
        elif self.config.mm_redundant_token_selection == "AvgPooling" or self.config.mm_redundant_token_selection == "MaxPooling":
            self.token_selection = kth_bipartite_soft_matching_pooling
            if self.config.mm_redundant_token_selection == "AvgPooling":
                self.token_selection_method = avg_pooling
            else:
                self.token_selection_method = max_pooling
            self.token_source = pooling_source if self.config.mm_token_source else None

class LLaMAVIDMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_token_selection(self):
        return self.get_model().token_selection
    def get_token_selection_method(self):
        return self.get_model().token_selection_method
    def get_token_source(self):
        return self.get_model().token_source
    def get_text_token_pruning(self):
        return self.get_model().text_model, self.get_model().text_tokenizer, self.get_model().text_token_pruning

    #changed function definition
    #def encode_images(self, images, prompts=None, image_counts=None, long_video=False):
    def encode_images(self, images, long_video=False, prompts=None, labels=None, top_k=1000):
        image_features = self.get_model().get_vision_tower()(images)
        # remove cls embedding
        if self.config.mm_vision_select_feature == 'patch':
            if image_features.shape[1] % 2 == 1:
                image_features = image_features[..., :, 1:]
        ndim = image_features.ndim
        if ndim == 3:
            image_features = image_features.unsqueeze(0)
        image_features = image_features.reshape(image_features.shape[0], image_features.shape[1]*image_features.shape[2], image_features.shape[3])

        if n_dim == 4:
            image_features = image_features.reshape(image_features.shape[-3]*image_features.shape[-2], image_features.shape[-1])
        else:
            image_features = image_features.reshape(image_features.shape[0], image_features.shape[-3]*image_features.shape[-2], image_features.shape[-1])

        # token merging
        if self.config.mm_redundant_token_selection:
            if images.shape[0] == 1: #TODO: WHAT IF THERE IS BATCH SIZE? print shape when training and exit
                kth = 1 #equivalent to doing nothing
            elif images.shape[i+1] < 96: # first third of the 5 minutes
                kth = 2
            elif images.shape[i+1] < 200: # second third of the 5 minutes
                kth = 4
            else:
                kth = 8 #kth is not dynamically chosen since all training videos are up to 5 minutes

            merge, _ = self.get_token_selection()(image_features, kth,
                                                  self.config.mm_token_merging_merge,
                                                  self.config.mm_token_merging_st_dist,
                                                  self.config.mm_lambda_t, self.config.mm_lambda_s)
            if self.config.mm_token_source:
                image_features_copy = image_features.clone()

            if self.config.mm_token_merging_merge:
                image_features, _ = self.get_token_selection_method()(merge, image_features)
            else:
                image_features = merge(image_features)

            if self.config.mm_token_source:
                image_features_source = self.get_token_source()(merge, image_features_copy)


        if self.config.mm_text_token_pruning:


            for p in range(len(prompts)):
                text_embeds = None
                sqz = False
                if image_features.dim() == 2:
                    image_features = image_features.unsqueeze(0)
                    sqz = True
                image_features_copy = None
                labels_copy = None
                if top_k[p] < image_features[p].shape[0]:
                    text_inputs = tokenizer(prompts[p], return_tensors="pt").to(device=self.device)
                    if text_embeds is None:
                        text_embeds = text_model(**text_inputs).last_hidden_state
                    else:
                        text_embeds = torch.cat((text_embeds, text_model(**text_inputs).last_hidden_state), dim=0)
                    if self.config.mm_token_source:  # TODO:  plotting should not be hard coded
                        plot_source_top_k_tokens(image_features[p], text_embeds, images, image_features_source)

                text_model, tokenizer, token_pruning = self.get_text_token_pruning()
                text_inputs = tokenizer(prompts, return_tensors="pt").to(device=self.device)
                text_embeds = text_model(**text_inputs).last_hidden_state[0]

        image_features = self.get_model().mm_projector(image_features)
        image_features = torch.squeeze(image_features, dim=0)
        image_features_list = [image_features]
        return image_features_list, labels

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


        # calculate the topk for text token pruning and the kth for token selection
        system_size = 
        text_size = input_ids.shape[1]
        top_k = max(0, self.config.max_position_embeddings - (system_size + text_size))

        if images.ndim() == 4 and images.shape[0] == 1:
            kth = 1  # equivalent to doing nothing



        if type(images) is list or images.ndim == 5:

            images = [image if len(image.shape) == 4 else image.unsqueeze(0) for image in images]
            concat_images = torch.cat(images, dim=0)
            image_features = self.encode_images(concat_images, prompts, labels, topk)
        else:
            image_features = self.encode_images(images, prompts, labels, topk)
        


        if type(images) is list or images.ndim == 5:
            # not reseshape for long video
            if not long_video:
                images = [image if len(image.shape) == 4 else image.unsqueeze(0) for image in images]
            concat_images = torch.cat(images, dim=0)
            if labels is not None:
                image_features, labels = self.encode_images(concat_images, long_video=long_video, prompts=prompts, labels=labels, top_k=top_k)
            else:
                image_features, _ = self.encode_images(concat_images, long_video=long_video, prompts=prompts, top_k=top_k)
        else:
            if labels is not None:
                image_features, labels = self.encode_images(images, long_video=long_video, prompts=prompts,
                                                            labels=labels, top_k=top_k)
            else:
                image_features, _ = self.encode_images(images, long_video=long_video, prompts=prompts, top_k=top_k)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        #cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                if isinstance(image_features, list):
                    #cur_image_features = image_features[cur_image_idx][0]
                    cur_image_features = image_features[0]
                else:
                    #cur_image_features = image_features[cur_image_idx]
                    cur_image_features = image_features
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                #cur_image_idx += 1
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
                        #cur_image_features = image_features[cur_image_idx][0]
                        cur_image_features = image_features[0]
                    else:
                        #cur_image_features = image_features[cur_image_idx]
                        cur_image_features = image_features
                    image_token_start = image_token_indices[0]
                    
                    if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                        if labels is not None:
                            cur_new_labels.append(cur_labels[:image_token_start])
                            cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype)) #AQUI? as labels são criadas com a shape das cur_im_features
                            cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                            cur_labels = cur_labels[image_token_start+2:]
                    else:
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                        cur_new_input_embeds.append(cur_image_features)
                        if labels is not None:
                            cur_new_labels.append(cur_labels[:image_token_start])
                            cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype)) #AQUI? as labels são criadas com a shape das cur_im_features
                            cur_labels = cur_labels[image_token_start+1:]
                    if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                        cur_input_ids = cur_input_ids[image_token_start+2:]
                    else:
                        cur_input_ids = cur_input_ids[image_token_start+1:]
                    image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

                
                # changle image idx after processing one sample
                #cur_image_idx += 1
                if cur_input_ids.numel() > 0:
                    if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
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
                cur_new_input_embeds = torch.Tensor(len(cur_input_ids), self.config.hidden_size).to(dtype=self.dtype, device=self.device)
                text_token_indices = torch.where(cur_input_ids != IMAGE_TOKEN_INDEX)[0]
                if not self.training and self.get_model().embed_tokens.weight.device != cur_input_ids.device:
                    model_device = self.get_model().embed_tokens.weight.device
                    data_device = cur_input_ids.device
                    cur_input_ids_text = cur_input_ids[text_token_indices].to(device=model_device)
                    cur_new_input_embeds[text_token_indices] = self.get_model().embed_tokens(cur_input_ids_text).to(device=data_device)
                else:
                    cur_new_input_embeds[text_token_indices] = self.get_model().embed_tokens(cur_input_ids[text_token_indices])
                #cur_image_features = image_features[cur_image_idx]
                cur_image_features = image_features
                cur_new_input_embeds[image_token_indices] = cur_image_features
                new_input_embeds.append(cur_new_input_embeds)
                if labels is not None:
                    new_labels.append(cur_labels)
                #cur_image_idx += 1

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            # only used for right padding in tokenlizer
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            # only used for right padding in tokenlizer
            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
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
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
