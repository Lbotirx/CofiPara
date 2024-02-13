import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

from models.t5.t5_model import T5ForConditionalGeneration
from models.groundingdino.util.inference import load_model, load_image, predict, annotate
from models.groundingdino.models.GroundingDINO.fuse_modules import FeatureResizer

from transformers import (
    AdamW,
    T5Config,
    RobertaConfig,
    T5TokenizerFast,
    RobertaModel,
    RobertaTokenizerFast,
    get_linear_schedule_with_warmup)
from models.util.slconfig import SLConfig
from utils import util
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions,SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from tqdm import tqdm
import socket
from collections import OrderedDict
from typing import *
from utils.loadData import *


class CofiPara(torch.nn.Module):
    def __init__(self, args, params = None, **kwargs) -> None:
        super(CofiPara, self).__init__()
        assert args.model_type in ['t5','roberta']
        self.params = params
        self.args = args
        self.save_path = args.save_path

        # init dino model
        dino_config = "models/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        ckpt = args.g_dino_ckpt_path
        self.dino = load_model(dino_config,ckpt)
        self.proj_layer = nn.Linear(256, 768, bias=True)
        
        self.softmax = torch.nn.Softmax(dim=1)

        self.model_type = args.model_type

        # init t5 model
        # if not using checkpoint, use pretrained t5
        if self.model_type == 't5':
            if not args.checkpoint:
                self.text_model = T5ForConditionalGeneration.from_pretrained(args.model_path)
            else:
                config = T5Config.from_json_file(args.model_path+'/config.json')
                self.text_model = T5ForConditionalGeneration(config=config)

            self.tokenizer = T5TokenizerFast.from_pretrained(args.model_path)

        if args.checkpoint is not None:
            print('Loading form checkpoint:',args.model_path)
            self.load_from_state_dict(args)
    

    def forward(self, text = None, image = None, input_ids = None,attention_mask = None,labels=None, return_dict=True, src_out = None):
        # encode text
        if input_ids is None:
            input_ids = self.tokenizer.encode(text, return_tensors="pt")
            encoder_output = self.text_model.encoder(input_ids = input_ids.cuda())

        if self.model_type == 't5':
            encoder_output = self.text_model.encoder(input_ids = input_ids,
                                                        attention_mask = attention_mask)
        else:
            encoder_output = self.text_model(input_ids = input_ids,
                                                attention_mask=attention_mask)
        encoded_text_feature = encoder_output[0]
        
        srcs, masks, text_dict, memory, memory_text = self.dino.cuda().feat_enhance_and_detr_enc(image.cuda(),
                                                                                                 encoded_text_feature.cuda(),
                                                                                                 src_out.to(self.args.device))

        # bbox decoding
        if self.args.stage == "stage2":
            img_tgt_dict = self.dino(image.cuda(),
                                     encoded_text_feature.cuda(),
                                     src_out.to(self.args.device))
        else:
            img_tgt_dict = None
            
        memory_text = self.proj_layer(memory_text)
        memory_text = encoded_text_feature + memory_text

        # text decoding with co-attended features
        if self.model_type == 't5':        
            encoder_output = BaseModelOutputWithPastAndCrossAttentions(last_hidden_state = memory_text.cuda())
            outputs = self.text_model.generate(
                                # input_ids = decoder_input,
                                encoder_outputs = encoder_output,
                                attention_mask = attention_mask,
                                labels = labels,
                                temperature = 0.5,
                                return_dict = return_dict,
                                image_features = memory
                                )
            # text_output = self.tokenizer.batch_decode(outputs,skip_special_tokens=True)
        else:
            logits = self.classifier(memory_text.cuda())

            outputs =  logits.argmax(dim=1)

        text_output = outputs

        return text_output, img_tgt_dict

    def forward_train(self, text = None, image = None, input_ids = None,attention_mask = None,labels=None, return_dict=True, src_out = None):
        # encode text
        if input_ids is None:
            input_ids = self.tokenizer.encode(text, return_tensors="pt")
            encoder_output = self.text_model.encoder(input_ids = input_ids.cuda())

        if self.model_type == 't5':
            encoder_output = self.text_model.encoder(input_ids = input_ids,
                                                        attention_mask = attention_mask)
    
        else:
            encoder_output = self.text_model(input_ids = input_ids,
                                                attention_mask=attention_mask)
        encoded_text_feature = encoder_output[0]
        
        srcs, masks, text_dict, memory, memory_text = self.dino.cuda().feat_enhance_and_detr_enc(image.cuda(),
                                                                                                 encoded_text_feature.cuda(),
                                                                                                 src_out.to(self.args.device))

        # bbox decoding with dino
        if self.args.stage == "stage2":
            # img_tgt_dict = self.dino(srcs, masks, text_dict, memory, memory_text)
            img_tgt_dict = self.dino(image.cuda(),
                                     encoded_text_feature.cuda(),
                                     src_out.to(self.args.device))
        else:
            img_tgt_dict = None

        memory_text = self.proj_layer(memory_text)
        memory_text = encoded_text_feature + memory_text

        # text decoding with co-attended features
        if self.model_type == 't5':
            encoder_output = BaseModelOutputWithPastAndCrossAttentions(last_hidden_state = memory_text.cuda())
            outputs = self.text_model(
                                decoder_input_ids = None,
                                encoder_outputs = encoder_output,
                                labels = labels,
                                image_features = memory,
                                return_dict = return_dict
                                )
        else:
            logits = self.classifier(memory_text.cuda())
            loss = None
            if labels is not None:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            outputs =  SequenceClassifierOutput(
                loss=loss,
                logits=logits,
            )

        return outputs,img_tgt_dict
    
    def save_state_dict(self, args):
        # save all parameters
        save_name = self.save_path + args.exp_name +'.pth'
        print('saving to:',save_name)
        torch.save({'model':self.state_dict()},
                        save_name)

    def load_from_state_dict(self, args):
        save_name = args.checkpoint
        assert save_name is not None
        print('loading from:',save_name)
        ckpt = torch.load(save_name, map_location='cuda:0')
        # ckpt["model"].pop('dino.transformer.tgt_embed.weight')  # not needed
        self.load_state_dict(ckpt['model'],strict=False)
