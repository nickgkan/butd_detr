# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from transformers import RobertaModel, RobertaTokenizerFast
from torchvision.ops import RoIPool

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn
import ipdb
st = ipdb.set_trace


class BeautyDetrTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False,
        num_feature_levels=4,
        dec_n_points=4,
        enc_n_points=4,
        two_stage=False,
        two_stage_num_proposals=300,
        pass_pos_and_query=True,
        text_encoder_type="roberta-base",
        freeze_text_encoder=False,
        butd=False,
        butd_class_embeddings=None,
        with_learned_class_embeddings=False
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.pass_pos_and_query = pass_pos_and_query
        self.butd = butd
        encoder_layer = CrossEncoderLayer(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_feature_levels=num_feature_levels,
            n_heads=nhead,
            enc_n_points=enc_n_points,
            butd=self.butd
        )

        self.encoder = CrossEncoder(
            encoder_layer,
            num_encoder_layers
        )

        decoder_layer = DecoderLayer(d_model, dim_feedforward,
                                                        dropout, activation,
                                                        num_feature_levels, nhead, dec_n_points, butd=butd)
        self.decoder = Decoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

        # Text Encoder
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            text_encoder_type)
        self.text_encoder = RobertaModel.from_pretrained(
            text_encoder_type)

        if freeze_text_encoder:
            print("Freezing text encoder")
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.expander_dropout = 0.1
        config = self.text_encoder.config
        self.resizer = FeatureResizer(
            input_feat_size=config.hidden_size,
            output_feat_size=d_model,
            dropout=self.expander_dropout,
        )
        
        if self.butd:
            input_channel = 128
            self.box_mlp = nn.Sequential(
                nn.Linear(128, 128),
                nn.LayerNorm(128, eps=1e-12),
                nn.Dropout(0.1))
            if with_learned_class_embeddings:
                input_channel = input_channel + 768
            else:
                input_channel = input_channel + 32
            self.butd_box_embedding = nn.Sequential(
                nn.Linear(input_channel, d_model),
                nn.LayerNorm(d_model, eps=1e-12),
                nn.Dropout(0.1)
            )
            assert butd_class_embeddings is not None
            self.butd_class_embeddings = butd_class_embeddings

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals, num_pos_feats=128):
        temperature = 10000
        scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        proposals = proposals.sigmoid() * scale
        pos = proposals[:, :, :, None] / dim_t
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)  # 2 X 13400 X 4
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)  # 2, 17821, 4
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_roi_pool_boxes(self, butd_boxes):
        B, N, _ = butd_boxes.shape
        indices = torch.repeat_interleave(
            torch.arange(len(butd_boxes),
            dtype=butd_boxes.dtype,
            device=butd_boxes.device),
            N
        )[:, None]
        pool_boxes = torch.cat([indices,
                 butd_boxes.reshape(-1, 4)], dim=-1)
        return pool_boxes

    def forward(
        self,
        srcs=None,
        masks=None,
        pos_embeds=None,
        query_embed=None,
        text=None,
        encode_and_save=True,
        filled_memory_cache=None,
        butd_boxes=None,
        butd_classes=None,
        butd_masks=None,
    ):
        if encode_and_save:
            assert self.two_stage or query_embed is not None

            # prepare input for encoder
            # Flatten spatial maps from resnet and concatenate them along flattned dim
            src_flatten = []
            mask_flatten = []
            lvl_pos_embed_flatten = []
            spatial_shapes = []
            for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
                bs, c, h, w = src.shape
                spatial_shape = (h, w)
                spatial_shapes.append(spatial_shape)
                src = src.flatten(2).transpose(1, 2)  # (2, 256, 100, 134) -> [2, 13400, 256]
                mask = mask.flatten(1)  
                pos_embed = pos_embed.flatten(2).transpose(1, 2)  # [2, 13400, 256]
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
                lvl_pos_embed_flatten.append(lvl_pos_embed)
                src_flatten.append(src)
                mask_flatten.append(mask)
            src_flatten = torch.cat(src_flatten, 1)  # [2, 17821, 256]
            mask_flatten = torch.cat(mask_flatten, 1)
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)  # [4, 2]
            level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))  # [4]
            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)  # [2, 4, 2]
         
            # Encode Text
            device = src.device
            if isinstance(text[0], str):
                # Encode the text
                tokenized = self.tokenizer.batch_encode_plus(text, padding="longest", return_tensors="pt").to(device)
                encoded_text = self.text_encoder(**tokenized)
                text_memory = encoded_text.last_hidden_state.transpose(0, 1)

                # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
                text_attention_mask = tokenized.attention_mask.ne(1).bool()

                # Resize the encoder hidden states to be of the same d_model as the decoder
                text_memory_resized = self.resizer(text_memory)  # seq X B X 256

            else:
                # The text is already encoded, use as is.
                text_attention_mask, text_memory_resized, tokenized = text
            
            butd_box_embeddings = None
            butd_box_pos_encoding = None
            if butd_boxes is not None:
                butd_box_pos_encoding = self.box_mlp(
                        self.get_proposal_pos_embed(butd_boxes, 32)) # B, 32, 128
               
                if butd_classes is not None:
                    butd_class_embedding = self.butd_class_embeddings(butd_classes)
                    butd_box_pos_encoding = torch.cat([
                        butd_box_pos_encoding,
                        butd_class_embedding
                    ], dim=-1)
                butd_box_embeddings = self.butd_box_embedding(butd_box_pos_encoding)

            # encoder
            memory, text_memory = self.encoder(
                src=src_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                pos=lvl_pos_embed_flatten,
                padding_mask=mask_flatten,
                text_memory=text_memory_resized,
                text_memory_mask=text_attention_mask,
                detected_feats=butd_box_embeddings,
                detected_mask=butd_masks,
                detected_pos=None # only useful for feature exp
            )

            # prepare input for decoder
            bs, _, c = memory.shape
            if self.two_stage:
                output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
                class_embed = self.decoder.class_embed[self.decoder.num_layers]
                enc_outputs_class = class_embed(output_memory)
                enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

                topk = self.two_stage_num_proposals 
                topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
                topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
                topk_coords_unact = topk_coords_unact.detach() 
                reference_points = topk_coords_unact.sigmoid()
                init_reference_out = reference_points
                pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
                query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
            else:
                query_embed, tgt = torch.split(query_embed, c, dim=1)
                query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
                tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
                reference_points = self.reference_points(query_embed).sigmoid()
                init_reference_out = reference_points
                enc_outputs_class = None
                enc_outputs_coord_unact = None

            memory_cache = {
                "text_memory_resized": text_memory_resized,
                "text_memory": text_memory,
                "img_memory": memory,
                "mask": mask,
                "text_attention_mask": text_attention_mask,
                "pos_embed": pos_embed,
                "query_embed": query_embed,
                "tgt": tgt,
                "reference_points": reference_points,
                "init_reference_out": init_reference_out,
                "spatial_shapes": spatial_shapes,
                "level_start_index": level_start_index,
                "valid_ratios": valid_ratios,
                "mask_flatten": mask_flatten,
                "tokenized": tokenized,
                "enc_outputs_class": enc_outputs_class,
                "enc_outputs_coord_unact": enc_outputs_coord_unact,
                "butd_box_embeddings": butd_box_embeddings,
                "butd_box_pos_embedding": butd_box_pos_encoding,
            }
            return memory_cache
        else:
            tgt = filled_memory_cache["tgt"]
            reference_points = filled_memory_cache["reference_points"]
            memory = filled_memory_cache["img_memory"]
            spatial_shapes = filled_memory_cache["spatial_shapes"]
            level_start_index = filled_memory_cache["level_start_index"]
            valid_ratios = filled_memory_cache["valid_ratios"]
            query_embed = filled_memory_cache["query_embed"]
            mask_flatten = filled_memory_cache["mask_flatten"]
            text_memory = filled_memory_cache["text_memory"]
            text_memory_resized = filled_memory_cache["text_memory_resized"]
            text_attention_mask = filled_memory_cache["text_attention_mask"]
            init_reference_out = filled_memory_cache["init_reference_out"]
            enc_outputs_class = filled_memory_cache["enc_outputs_class"]
            enc_outputs_coord_unact = filled_memory_cache[
                "enc_outputs_coord_unact"]
            butd_box_embeddings = filled_memory_cache['butd_box_embeddings']
            butd_box_pos_embedding = filled_memory_cache['butd_box_pos_embedding']
            if butd_boxes is not None and butd_box_embeddings is None:
                butd_box_pos_encoding = self.box_mlp(
                    self.get_proposal_pos_embed(butd_boxes, 32)) # B, 32, 128
                if butd_classes is not None:
                    butd_class_embedding = self.butd_class_embeddings(butd_classes)
                    butd_box_pos_encoding = torch.cat([
                        butd_box_pos_encoding,
                        butd_class_embedding
                    ], dim=-1)
                butd_box_embeddings = self.butd_box_embedding(butd_box_pos_encoding)

            # decoder

            hs, inter_references = self.decoder(
                tgt=tgt,
                reference_points=reference_points,
                src=memory,
                src_spatial_shapes=spatial_shapes,
                src_level_start_index=level_start_index,
                src_valid_ratios=valid_ratios,
                query_pos=query_embed,
                src_padding_mask=mask_flatten,
                text_memory=text_memory,
                text_key_padding_mask=text_attention_mask,
                butd_box_embeddings=butd_box_embeddings,
                butd_masks=butd_masks,
                butd_box_pos_embedding=None # only useful for feature exp
            )

            inter_references_out = inter_references
            if self.two_stage:
                return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
            return hs, init_reference_out, inter_references_out, None, None


class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)  # [2, 17821, 256]

        return src

class CrossTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4
    ):
        super().__init__()

        #  cross attention from lang to vision
        self.cross_lv = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout
        )
        self.dropout_lv = nn.Dropout(dropout)
        self.norm_lv = nn.LayerNorm(d_model)

        # cross attention from vision to lang
        self.cross_vl = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout
        )
        self.dropout_vl = nn.Dropout(dropout)
        self.norm_vl = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        output_img=None,
        img_key_padding_mask=None,
        output_text=None,
        text_key_padding_mask=None,
    ):

        # produce key, query, value for image
        qv = kv = vv = output_img

        # produce key, query, value for text
        qt = kt = vt = output_text

        # cross attend language to vision
        output_text2 = self.cross_lv(
            query=qt,
            key=kv.transpose(0, 1),
            value=vv.transpose(0, 1),
            attn_mask=None,
            key_padding_mask=img_key_padding_mask,
        )[0]
        output_text = output_text + self.dropout_lv(output_text2)
        output_text = self.norm_lv(output_text)

        # cross attend image to language
        output_img2 = self.cross_vl(
            query=qv.transpose(0, 1),
            key=kt,
            value=vt,
            attn_mask=None,
            key_padding_mask=text_key_padding_mask,
        )[0].transpose(0, 1)
        output_img = output_img + self.dropout_vl(output_img2)
        output_img = self.norm_vl(output_img)

        return output_img, output_text

class CrossEncoder(nn.Module):
    def __init__(self, bi_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(bi_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)  # [100, 134] row same, column increment
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)  # [100, 134] row change, column same
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        src,
        pos,
        valid_ratios,
        spatial_shapes,
        level_start_index,
        padding_mask,
        text_memory,
        text_memory_mask,
        detected_feats=None,
        detected_mask=None,
        detected_pos=None
    ):
        output_image = src
        output_text = text_memory
        reference_points = self.get_reference_points(
            spatial_shapes,
            valid_ratios,
            device=src.device
        )
        for _, layer in enumerate(self.layers):
            output_image, output_text = layer(
                src=output_image,
                pos=pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                padding_mask=padding_mask,
                text_memory=output_text,
                text_memory_mask=text_memory_mask,
                detected_feats=detected_feats,
                detected_mask=detected_mask,
                detected_pos=detected_pos
            )

        return output_image, output_text

class CrossEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        dropout=0.1,
        activation="relu",
        n_heads=8,
        dim_feedforward=256,
        num_feature_levels=4,
        enc_n_points=4,
        butd=False
    ):
        super().__init__()

        # self attention in language
        self.self_attention_lang = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout
        )
        self.dropout_lang = nn.Dropout(dropout)
        self.norm_lang = nn.LayerNorm(d_model)

        # self attention in vision
        self.self_attention_visual = EncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            n_heads,
            enc_n_points
        )

        # cross attention in language and vision
        self.cross_layer = CrossTransformerEncoderLayer(
                d_model,
                dim_feedforward,
                dropout,
                activation,
                num_feature_levels,
                n_heads,
                enc_n_points,
            )
        
        # cross attention from language to detected boxes
        if butd:
            self.cross_d = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout
                )
            self.dropout_d = nn.Dropout(dropout)
            self.norm_d = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        padding_mask,
        text_memory,
        text_memory_mask,
        detected_feats=None,
        detected_mask=None,
        detected_pos=None
    ):

        # do self attention in image
        output_image_feats = src
        output_image_feats = self.self_attention_visual(
            output_image_feats,
            pos,
            reference_points,
            spatial_shapes,
            level_start_index,
            padding_mask
        )

        # do self attention in language
        output_text = text_memory
        qt = kt = vt = text_memory
        output_text2 = self.self_attention_lang(
            query=qt,
            key=kt,
            value=vt,
            attn_mask=None,
            key_padding_mask=text_memory_mask
        )[0]
        output_text = text_memory + self.dropout_lang(output_text2)
        output_text = self.norm_lang(output_text)

        # do cross attention
        output_image, output_text = self.cross_layer(
            output_img=output_image_feats,
            img_key_padding_mask=padding_mask,
            output_text=output_text,
            text_key_padding_mask=text_memory_mask,
        )
        
        if detected_feats is not None:
            if detected_pos is not None:
                detected_feats_key = detected_feats + detected_pos
            else:
                detected_feats_key = detected_feats
            output_image2 = self.cross_d(
                query=output_image.transpose(0, 1),
                key=detected_feats_key.transpose(0, 1),
                value=detected_feats.transpose(0, 1),
                attn_mask=None,
                key_padding_mask=~detected_mask
            )[0].transpose(0, 1)
            output_image = output_image + self.dropout_d(output_image2)
            output_image = self.norm_d(output_image)

        return output_image, output_text

class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                dropout=0.1, activation="relu",
                n_levels=4, n_heads=8, n_points=4, butd=False):
        super().__init__()

        # cross attention: deformable
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # cross attention: normal
        self.cross_attn_normal = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout
        )
        self.dropout_ca_normal = nn.Dropout(dropout)
        self.norm_ca_normal = nn.LayerNorm(d_model)
        
        # butd cross attention
        if butd:
            self.cross_d = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout
            )
            self.dropout_d = nn.Dropout(dropout)
            self.norm_d = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_spatial_shapes,
        level_start_index,
        src_padding_mask=None,
        text_memory=None,
        text_key_padding_mask=None,
        butd_box_embeddings=None,
        butd_masks=None,
        butd_box_pos_embedding=None
    ):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            tgt.transpose(0, 1)
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention with language
        tgt2 = self.cross_attn_normal(
            query=self.with_pos_embed(tgt, query_pos).transpose(0, 1),
            key=text_memory,
            value=text_memory,
            attn_mask=None,
            key_padding_mask=text_key_padding_mask,
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout_ca_normal(tgt2)
        tgt = self.norm_ca_normal(tgt)
        
        # cross attention with detected boxes
        if butd_box_embeddings is not None:
            if butd_box_pos_embedding is not None:
                butd_box_embeddings_key = butd_box_embeddings + butd_box_pos_embedding
            else:
                butd_box_embeddings_key = butd_box_embeddings
            tgt2 = self.cross_d(
                query=self.with_pos_embed(tgt, query_pos).transpose(0, 1),
                key=butd_box_embeddings_key.transpose(0, 1),
                value=butd_box_embeddings.transpose(0, 1),
                key_padding_mask=~butd_masks
            )[0].transpose(0, 1)
            tgt = tgt + self.dropout_d(tgt2)
            tgt = self.norm_d(tgt)

        # cross attention with image
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(
        self,
        tgt,
        reference_points,
        src,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        query_pos=None,
        src_padding_mask=None,
        text_memory=None,
        text_key_padding_mask=None,
        butd_box_embeddings=None,
        butd_masks=None,
        butd_box_pos_embedding=None
    ):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):  # 6
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]  # 2, 300, 4, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] \
                    * src_valid_ratios[:, None]
            output = layer(
                output,
                query_pos,
                reference_points_input,
                src,
                src_spatial_shapes,
                src_level_start_index,
                src_padding_mask,
                text_memory=text_memory,
                text_key_padding_mask=text_key_padding_mask,
                butd_box_embeddings=butd_box_embeddings,
                butd_masks=butd_masks,
                butd_box_pos_embedding=butd_box_pos_embedding
            )

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_beauty_detr_transformer(args, butd_class_embeddings):
    return BeautyDetrTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        text_encoder_type=args.text_encoder_type,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        butd=args.butd,
        butd_class_embeddings=butd_class_embeddings,
        with_learned_class_embeddings=args.with_learned_class_embeddings
    )
