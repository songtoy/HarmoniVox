# Portions adapted from Salesforce LAVIS BLIP-2; Copyright (c) 2023, salesforce.com, inc.
# Upstream terms: https://github.com/salesforce/LAVIS/blob/main/LICENSE.txt
# HarmoniVox research adaptations are recorded in docs/CODE_ORIGINS.md.
# Attribution notice added 2026-11-25; no model computation changed.

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

import logging
from blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)

from blip2_models.base_model import all_gather_with_grad, concat_all_gather
from models import *
from transformers import AutoTokenizer
from release_assets import qformer_path



class Blip2Qformer(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """


    def __init__(
        self,
        vit_model,
        freeze_vit=True,
        num_query_token=32,
        audio_embed_dim=512,
        cross_attention_freq=2,
        embed_dim=512,
        max_txt_len=32,
        temperature=0.07,
        qformer_config=None,
    ):
        super().__init__()

        #self.tokenizer = self.init_tokenizer()

        assert vit_model is not None, "vit_model is required"

        vit = vit_model.visual
        vit.output_tokens = True
        vit.proj = None
        self.visual_encoder = vit
        self.visual_encoder.num_features = 1280

        self.ln_vision = nn.LayerNorm(self.visual_encoder.num_features)

        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            print("freeze vision encoder")

        # The public image-only path returns before the auxiliary text branch.
        # Keep that branch's tokenizer lazy so inference needs no XLM-R assets.
        self.tokenizer = None

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq,
            config_path=qformer_config,
        )
        #self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])


        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.audio_proj = nn.Linear(audio_embed_dim, embed_dim) if audio_embed_dim != embed_dim else nn.Identity()
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.modal_fusion = nn.Linear(embed_dim * 2, embed_dim)

        self.itm_classifier = AdversarialClassifier(embed_dim, out_dim=2)  # match or not match
        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        self.itm_head.weight.data.normal_(mean=0.0, std=0.02)
        self.itm_head.bias.data.zero_()
        

        print("use temperature: ", temperature)
        self.temp = nn.Parameter(temperature * torch.ones([]))

        self.max_txt_len = max_txt_len

        
    def forward_image_audio(self, image_feats_raw, audio_feat_raw):
        device = image_feats_raw.device
        image_feats = F.normalize(image_feats_raw, dim=-1)
        
        audio_feat = F.normalize(audio_feat_raw, dim=-1)
        
        ###============== Image-text Contrastive ===================###
        image_feats_all = image_feats #image_feats_all = concat_all_gather(image_feats)  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        audio_feat_all = audio_feat  #audio_feat_all = concat_all_gather(audio_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            image_feats.unsqueeze(1), audio_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            audio_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        #rank = dist.get_rank()
        bs = image_feats_raw.size(0)
        targets = torch.linspace(0, bs - 1, bs, dtype=int).to( #targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            device
        )

        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        ###============== Image-text Matching ===================###

        audio_embeds_world = audio_feat_raw
        image_embeds_world = image_feats_raw

        with torch.no_grad():
            sim_t2i[:, : bs].fill_diagonal_(-10000)
            sim_i2t[:, : bs].fill_diagonal_(-10000)            
                
            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_i2t = F.softmax(sim_i2t, dim=1)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        audio_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            audio_embeds_neg.append(audio_embeds_world[neg_idx])
        audio_embeds_neg = torch.stack(audio_embeds_neg, dim=0)

        audio_embeds_all = torch.cat(
            [audio_feat_raw, audio_feat_raw, audio_embeds_neg], dim=0
        )  # pos, pos, neg

        image_embeds_all = torch.cat(
            [image_feats_raw, image_embeds_neg, image_feats_raw], dim=0
        )  # pos, neg, pos
        
        # audio_embeds_all: 3B x C 
        # image_embeds_all: 3B x 32 x C
        fusion_embeds = torch.cat([audio_embeds_all.unsqueeze(1).expand(-1, image_embeds_all.shape[1], -1), image_embeds_all], dim=-1)

        fused_embeds = self.modal_fusion(fusion_embeds)

        fused_output = self.itm_classifier(fused_embeds, is_reversal=False)

        logits = fused_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(device)
        loss_itm = F.cross_entropy(logits, itm_labels)

        return {
            "loss": loss_itc + loss_itm,# + loss_lm,
            "loss_itc": loss_itc,
            "loss_itm": loss_itm,
            #"loss_lm": loss_lm,
        }

    def forward(self, image, text=None, audio_style=None):
        device = image.device
            
        ## speech style-related visual representation
        # concat all tokens
        img_cls, img_tokens = self.visual_encoder(image)
        image_embeds = self.ln_vision(torch.cat([img_cls.unsqueeze(1), img_tokens], dim=1))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats_raw = self.vision_proj(query_output.last_hidden_state)

        if text is None or audio_style is None:
            image_feats = image_feats_raw.mean(dim=1) 
            image_feats = F.normalize(image_feats, dim=-1)
            return image_feats

        image_feats = F.normalize(image_feats_raw, dim=-1)

        audio_feat_raw = self.audio_proj(audio_style)
        audio_feat = F.normalize(audio_feat_raw, dim=-1)

        # extract text information
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(str(qformer_path()), local_files_only=True)
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)
        
        text_output = self.Qformer(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        ###============== Audio-Text Contrastive ===================###
        audio_feat_all = audio_feat
        text_feat_all = text_feat  #audio_feat_all = concat_all_gather(audio_feat)  # [batch_size*num_gpu, embed_dim]

        # jx1xnxn @ kxnxn -> jxkxnxn
        sim_a2t = torch.matmul(
            audio_feat.unsqueeze(1).unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        sim_a2t = sim_a2t / self.temp

        # text-audio similarity: [batch_size, batch_size*num_gpu]
        sim_t2a = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), audio_feat_all.unsqueeze(-1)
        ).squeeze()

        sim_t2a = sim_t2a / self.temp  # [batch_size, batch_size*num_gpu]

        #rank = dist.get_rank()
        bs = image.size(0)
        targets = torch.linspace(0, bs - 1, bs, dtype=int).to( #targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            device
        )

        loss_atc = (
            F.cross_entropy(sim_a2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2a, targets, label_smoothing=0.1)
        ) / 2

        ###============== Image-Text Contrastive ===================###
        image_feats_all = image_feats #image_feats_all = concat_all_gather(image_feats)  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = text_feat  #audio_feat_all = concat_all_gather(audio_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        #rank = dist.get_rank()
        bs = image.size(0)
        targets = torch.linspace(0, bs - 1, bs, dtype=int).to( #targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            device
        )

        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        ###============== Image-text Matching ===================###

        text_input_ids_world = text_tokens.input_ids
        text_attention_mask_world = text_tokens.attention_mask
        image_embeds_world = image_embeds

        with torch.no_grad():
            sim_t2i[:, : bs].fill_diagonal_(-10000)
            sim_i2t[:, : bs].fill_diagonal_(-10000)            
                
            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_i2t = F.softmax(sim_i2t, dim=1)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        image_embeds_all = torch.cat(
            [image_embeds, image_embeds_neg, image_embeds], dim=0
        )  # pos, neg, pos
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
            device
        )

        output_itm = self.Qformer(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(device)
        loss_itm = F.cross_entropy(logits, itm_labels)

        #fused_embeds = self.modal_fusion(fusion_embeds)

        #fused_output = self.itm_classifier(fused_embeds, is_reversal=False)

        #logits = fused_output.mean(dim=1)

        losses_image_audio = self.forward_image_audio(image_feats_raw, audio_feat_raw)
        

        return {
            "loss_atc": loss_atc,
            "loss_itc": loss_itc,
            "loss_itm": loss_itm,
            "loss_iac": losses_image_audio["loss_itc"],
            "loss_iam": losses_image_audio["loss_itm"],
        }

    def generate(self, image):
        device = image.device

        img_cls, img_tokens = self.visual_encoder(image)
        image_embeds = self.ln_vision(torch.cat([img_cls.unsqueeze(1), img_tokens], dim=1))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats_raw = self.vision_proj(query_output.last_hidden_state) 

        image_feats = image_feats_raw.mean(dim=1) 
        # image_feats = image_feats_raw[:, 0]
        # image_feats = image_feats_raw @ coef # coef: 32 x 1

        image_feats = F.normalize(image_feats, dim=-1)
        return image_feats
