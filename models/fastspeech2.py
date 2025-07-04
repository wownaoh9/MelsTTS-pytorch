import os
import json

import torch
import torch.nn as nn

# from .conformer import TextEncoder, Decoder
from .modules import SpeakerEncoder, LangEncoder, MelLinear, VarianceAdaptor, ResidualCouplingBlock, CoarseEmoExtractionModule, FineEmoExtractionModule, FineEmoPredictModule, PhoneLevelEmb, UtterLevelEmb
from transformer import Encoder, Decoder, PostNet

from .commons import sequence_mask
from utils.tools import get_mask_from_lengths

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        # FS2 Init
        self.model_config = model_config
        self.text_encoder = Encoder(model_config)
        self.speaker_encoder = SpeakerEncoder(preprocess_config, model_config)
        self.language_encoder = LangEncoder(preprocess_config, model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = MelLinear(model_config["conformer"]["decoder_hidden"],preprocess_config["preprocessing"]["mel"]["n_mel_channels"])
        self.spk_lang_linear = nn.Linear(64, 256)
        self.postnet = PostNet()
        
        #emo init
        self.coarse_emo_extraction_module = CoarseEmoExtractionModule(preprocess_config, model_config)




        
    def forward(            
            self,
            srcs, src_len, max_src_len,
            speaker, emotion_emo, lang_id,
            mels = None, mel_len = None, max_mel_len = None,
            pitch = None, energy = None, duration = None,

            #using for emotion encoder
            p_control=1.0,
            e_control=1.0,
            d_control=1.0,

            train = True
        ):
        device = srcs.device
        src_mask = get_mask_from_lengths(src_len, max_src_len, device)
        mel_mask = (
            get_mask_from_lengths(mel_len, max_mel_len, device)
            if mel_len is not None
            else None
        )

        # text encoder
        output = self.text_encoder(srcs, src_mask)
        
        # add speaker embedding
        speaker_emb = self.speaker_encoder(speaker)
        output = output + self.spk_lang_linear(speaker_emb).unsqueeze(1).repeat(1, output.shape[1], 1)

        # add language embedding
        lang_emb = self.language_encoder(lang_id)
        output = output + self.spk_lang_linear(lang_emb).unsqueeze(1).repeat(1, output.shape[1], 1)

        # coarse emo extraction
        coarse_emo_emb, cls_utter = self.coarse_emo_extraction_module(mels, mel_len, emotion_emo, speaker_emb, lang_emb)
        
        output = output + coarse_emo_emb.unsqueeze(1).repeat(1, output.size(1), 1)


        #variance adapter
        (   output,
            p_predictions, e_predictions, log_d_predictions, d_rounded,
            mel_lens, mel_mask
        ) = self.variance_adaptor(
            output, 
            src_mask,
            mel_mask, max_mel_len,
            pitch, energy, duration,
            p_control, e_control, d_control,
            train = train
        )

        #six conformer decoder to compute mel_loss seperately
        output, mel_masks= self.decoder(output, mel_mask)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions, e_predictions, log_d_predictions, d_rounded,
            src_mask, mel_mask, src_len, mel_lens,       #9
        )


    def infer(            
            self,
            srcs, src_len, max_src_len,
            speaker, emotion_emo, lang_id,
            mels = None, mel_len = None, max_mel_len = None,
            pitch = None, energy = None, duration = None,

            #using for emotion encoder
            p_control=1.0,
            e_control=1.0,
            d_control=1.0,

            train = True
        ):
        device = srcs.device
        src_mask = get_mask_from_lengths(src_len, max_src_len, device)
        mel_mask = (
            get_mask_from_lengths(mel_len, max_mel_len, device)
            if mel_len is not None
            else None
        )

        # text encoder
        output = self.text_encoder(srcs, src_mask)
        
        # add speaker embedding
        speaker_emb = self.speaker_encoder(speaker)
        output = output + self.spk_lang_linear(speaker_emb).unsqueeze(1).repeat(1, output.shape[1], 1)

        # add language embedding
        lang_emb = self.language_encoder(lang_id)
        output = output + self.spk_lang_linear(lang_emb).unsqueeze(1).repeat(1, output.shape[1], 1)

        # coarse emo extraction
        coarse_emo_emb, cls_utter = self.coarse_emo_extraction_module(mels, mel_len, emotion_emo, speaker_emb, lang_emb)
        
        output = output + coarse_emo_emb.unsqueeze(1).repeat(1, output.size(1), 1)

        
        #variance adapter
        (   output,
            p_predictions, e_predictions, log_d_predictions, d_rounded,
            mel_lens, mel_mask
        ) = self.variance_adaptor(
            output, 
            src_mask,
            mel_mask, max_mel_len,
            pitch, energy, duration,
            p_control, e_control, d_control,
            train = train
        )

        #six conformer decoder to compute mel_loss seperately
        output, mel_masks= self.decoder(output, mel_mask)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions, e_predictions, log_d_predictions, d_rounded,
            src_mask, mel_mask, src_len, mel_lens,       #9
        )