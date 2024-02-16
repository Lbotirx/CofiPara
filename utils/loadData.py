import numpy as np
from PIL import Image
import random
import pickle
import re
from torch.utils.data import Dataset, DataLoader
from typing import *
from transformers import T5TokenizerFast,RobertaTokenizerFast
import torch
from .prompt import prompt_train, prompt_test, prompt_expl, prompt_stage2,prompt_roberta
from PIL import Image
from models.transforms import Compose,RandomResize,ToTensor,Normalize,Padding,Resize
from tqdm import tqdm
import torch.nn.functional as F
from .eval import giou_losses
import json
from utils.eval import loss_by_feat_single,bbox_cxcywh_to_xyxy

loss_func = giou_losses


# Class To Load MSD data
wordPrefix = "{path_to_MSD_data}/MSD/extract/"
dataPrefix = "{path_to_MSD_data}/MSD/text/"
imagePrefix = "{path_to_MSD_data}/MSD/imageVector2/"

class TextItem():
    def __init__(self, sentence, label):
        self.sentence = sentence
        self.label = label
        self.words = None

class TextIterator():
    def __init__(self, batchSize = 32, seqLen = 75):
        self.batchSize = batchSize
        self.seqLen = seqLen
        self.textData = dict() 
        self.trainNum = []
        self.validNum = []
        self.testNum = []
        dictExtractWords = self.getExtractDict()
        for i in range(3):
            self.readData(i, dictExtractWords)
        self.batchInd = 0
        self.validInd = 0
        self.testInd = 0
        self.epoch = 0
        self.threshold = int(len(self.trainNum) / self.batchSize)
        # print('train set:',len(self.trainNum),'vaild set:', len(self.validNum),'test set:', len(self.testNum))
        # print("rate: ", self.rate)

    def getExtractDict(self):
        file = open(wordPrefix+"extract_all")
        dic = {}
        for line in file:
            ls = eval(line)
            dic[int(ls[0])] = ls[1:]
        return dic

    def readData(self, i, dic):
        p = n = 0
        if i == 0:
            file = open(dataPrefix+"train.txt")
            ls = self.trainNum
        elif i == 1:
            file = open(dataPrefix+"valid2.txt")
            ls = self.validNum
        else:
            file = open(dataPrefix+"test2.txt")
            ls = self.testNum
        for line in file:
            lineLS = eval(line)
            tmpLS = lineLS[1].split()
            if "sarcasm" in tmpLS:
                continue
            if "sarcastic" in tmpLS:
                continue
            if "reposting" in tmpLS:
                continue
            if "<url>" in tmpLS:
                continue
            if "joke" in tmpLS:
                continue
            if "humour" in tmpLS:
                continue
            if "humor" in tmpLS:
                continue
            if "jokes" in tmpLS:
                continue
            if "irony" in tmpLS:
                continue
            if "ironic" in tmpLS:
                continue
            if "exgag" in tmpLS:
                continue
            assert int(lineLS[0]) not in self.textData
            ls.append(int(lineLS[0]))
            if i == 0:
                if lineLS[-1] == 1:
                    p += 1
                else:
                    n += 1
            self.textData[int(lineLS[0])] = TextItem(lineLS[1], int(lineLS[-1]))
            self.textData[int(lineLS[0])].words = dic[int(lineLS[0])]
        random.shuffle(ls)
        if i == 0:
            self.rate = float(n) / p

    def getTestData(self):
        text_data = []
        image_data = []
        id_list = []
        for id in self.testNum:
            text = self.textData[id].sentence
            label = self.textData[id].label
            img_path = '{path_to_MSD_data}/MSD/dataset_image/' + str(id) +'.jpg'
            text_data.append(text)
            image_data.append(img_path)
            id_list.append(id)
        return text_data, image_data, id_list

    def getTrainData(self):
        text_data = []
        image_data = []
        id_list = []
        for id in self.trainNum:
            text = self.textData[id].sentence
            label = self.textData[id].label
            img_path = '{path_to_MSD_data}/MSD/dataset_image/' + str(id) +'.jpg'
            text_data.append(text)
            image_data.append(img_path)
            id_list.append(id)
        return text_data, image_data, id_list
    
    def getValidData(self):
        text_data = []
        image_data = []
        id_list = []
        for id in self.validNum:
            text = self.textData[id].sentence
            label = self.textData[id].label
            img_path = '{path_to_MSD_data}/MSD/dataset_image/' + str(id) +'.jpg'
            text_data.append(text)
            image_data.append(img_path)
            id_list.append(id)
        return text_data, image_data, id_list
    
def getScore(p, y):
    tp = fp = tn = fn = 0
    for i in range(len(p)):
        if y[i] == 1:
            if p[i] == 1:
                tp += 1
            else:
                fn += 1
        else:
            if p[i] == 1:
                fp += 1
            else:
                tn += 1
    return tp, fp, tn, fn

def getF1(tp, fp, tn, fn):
    try:
        pre = float(tp) / (tp+fp)
        rec = float(tp) / (tp+fn)
        f1 = 2*pre*rec / (pre+rec)
    except:
        pre = rec = f1 = 0
    return pre, rec, f1

import os
def load_sentence(tweet_data_dir, tpye_path):
    """
    read the word from doc, and build sentence. every line contain a word and it's tag
    every sentence is split with a empty line. every sentence begain with an "IMGID:num"

    """
    
    IMAGEID='IMGID'
    img_id = []
    sentences = []
    sentence = []
    datasplit = []
    
    assert tpye_path in ['train', 'val', 'test']

    datasplit.append(len(img_id))
    with open(os.path.join(tweet_data_dir, tpye_path), 'r', encoding='utf-8') as file:
        last_line = ''
        for line in file:
            line = line.rstrip()
            if line == '':
                sentences.append(sentence)
                sentence = []
            else:
                if IMAGEID in line:
                    num = line[6:]
                    # num = f"{path_to_MSD_data}/MSTI/datasets/img/{num}.jpg"
                    img_id.append(num)
                    if last_line != '':
                        print(num)
                else:
                    if len(line.split()) == 1:
                        print(line)
                    sentence.append(line.split())
            last_line = line

    targets = []
    out_sentences = []
    for sentence in sentences:
        target = []
        for word in sentence:
            try:
                if word[1] != 'O':
                    target.append(word[0])
            except IndexError:
                print(sentence)
        sen = ' '.join([word[0] for word in sentence])
        out_sentences.append(sen)
        targets.append(target)

    return out_sentences, targets, img_id, sentences


class T5DataSet(Dataset):
    def __init__(self, type_path, tokenizer = None, max_examples=-1,
                 max_src_len=512, max_tgt_len=200):
        """
        max_examples: if > 0 then will load only max_examples into the dataset; -1 means use all

        max_src and max_tgt len refer to number of tokens in the input sequences
        # Note: these are not randomized. If they were we might need to collate.
        """
        self.max_examples = max_examples
        self.max_src_len = max_src_len  # max num of tokens in tokenize()
        self.max_tgt_len = max_tgt_len
        self.use_rationale = False      # aborted!! distil step by step setting. 

        # self.data_path = data_dir
        self.type_path = type_path
        print('loading',type_path,'set')
        if self.type_path == 'test':
            import json 
            rationale_path0 = '{path_to_pos_rationale}_test.json'
            rationale_path1 = '{path_to_neg_rationale}_test.json'
            with open(rationale_path0,'r') as f:
                self.rationale0 = json.load(f)
            with open(rationale_path1,'r') as f:
                self.rationale1 = json.load(f)
        elif self.type_path == 'train':
            rationale_path0 = "{path_to_pos_rationale}_train.json"
            rationale_path1 = "{path_to_neg_rationale}_train.json"
            import json
            with open(rationale_path0,'r') as f:
                self.rationale0 = json.load(f)
            with open(rationale_path1,'r') as f:
                self.rationale1 = json.load(f)

        self.text_indicator = None
        # with open('mmsd_annotation.json','r') as f:
        #     self.text_indicator = json.load(f)
        
        self.texti = TextIterator()

        assert type_path in ['train','test','val']

        if tokenizer is None:
            self.tokenizer = T5TokenizerFast.from_pretrained('flan-t5-base')
        else:
            self.tokenizer  = tokenizer

        self.mmsd = 2.0
        if self.mmsd == 2.0:
            import json
            with open(f'./data/mmsd2/{type_path}_text.json','r') as f:
                self.mmsd_data = json.load(f)
            with open('./data/mmsd2/mmsd_label.json','r') as f:
                self.mmsd_label = json.load(f)

        self._build()       # fill inputs, targets, max_lens
        print('done loading',type_path,'set with length of',len(self.img_data))

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):
        src_text = self.source_data[index]
        tgt_text = self.target_text[index]
        image_id = self.img_data[index]

        # For distil-step-by-step setting
        if self.use_rationale:
            source_expl = self.source_expl[index]
            target_expl = self.target_expl[index]
            return {"source_text": src_text, "target_text": tgt_text,
                    "image_ids": image_id, "source_expl": source_expl,
                    "target_expl": target_expl}

        else:
            source_expl,target_expl = [], []
            return {"source_text": src_text, "target_text": tgt_text,
                    "image_ids": image_id}
            
        

    def _build(self):

        print("buliding dataset...")

        source_expl = []    # for rationale distillation
        target_expl = []
        self.source_data = []

        if self.type_path == 'train':
            
            self.use_rationale = False

            self.text_data, self.img_data, self.id_list = self.texti.getTrainData()

            
            for idx in range(len(self.id_list)):
                label = self.texti.textData[self.id_list[idx]].label
                data_id = self.id_list[idx]

                if self.mmsd == 2.0:
                    text = self.mmsd_data[str(data_id)]
                    label = self.mmsd_label[str(data_id)]
                else:
                    text = self.texti.textData[data_id].sentence.strip()

                # # to avoid shortcut
                # rationale0 = eval('self.rationale'+str(label))[str(data_id)].strip()
                # rationale1 = eval('self.rationale'+str(1-label))[str(data_id)].strip()

                rationale0, rationale1 = self.rationale0[str(data_id)].strip(),self.rationale1[str(data_id)].strip()

                text = prompt_train.format(text,
                                    rationale1,
                                    rationale0)
                # if self.text_indicator is not None:
                #     if self.text_indicator[str(data_id)]:
                #         text = text.replace("[label]","[label][text]")

                self.source_data.append(text)
                
                # text = prompt_test.format(self.text_data[idx].strip())          # ablation study without rationale
        else:
            if self.type_path == 'test':

                self.text_data, self.img_data, self.id_list = self.texti.getTestData()
                for idx in range(len(self.id_list)):
                    label = self.texti.textData[self.id_list[idx]].label
                    data_id = self.id_list[idx]
                    if self.mmsd == 2.0:
                        text = self.mmsd_data[str(data_id)]
                        label = self.mmsd_label[str(data_id)]
                    else:
                        text = self.texti.textData[data_id].sentence.strip()
                    rationale0, rationale1 = self.rationale0[str(data_id)].strip(),self.rationale1[str(data_id)].strip()           

                    text = prompt_train.format(text,
                                        rationale1,
                                        rationale0)
                    # if self.text_indicator is not None:
                    #     if self.text_indicator[str(data_id)]:
                    #         text = text.replace("[label]","[label][text]")
                    self.source_data.append(text)
                
                if self.use_rationale:
                    expl_prompt = prompt_expl.format(self.rationale0[str(data_id)].strip())
                    source_expl.append(expl_prompt)
                    target_expl.append(self.rationale0[str(data_id)].strip())
            else:
                self.text_data, self.img_data, self.id_list = self.texti.getValidData()

                for idx in range(len(self.id_list)):
                    label = self.texti.textData[self.id_list[idx]].label
                    data_id = self.id_list[idx]
                    text = prompt_test.format(self.text_data[idx].strip())
                    self.source_data.append(text)

        if self.use_rationale:
            self.source_expl = source_expl
            self.target_expl = target_expl
        else:
            self.source_expl,self.target_expl = [], []

        target = []
        targetdict = {0:'No',1:'Yes'}
        if self.mmsd == 2.0:
            for idx in range(len(self.id_list)):
                target.append(targetdict[self.mmsd_label[str(self.id_list[idx])]])        
        else:
            for idx in range(len(self.id_list)):
                target.append(targetdict[self.texti.textData[self.id_list[idx]].label])        
        self.target_text = target

class MSData(Dataset):
    def __init__(self,type,text_path = dataPrefix, img_path=imagePrefix) -> None:
        super().__init__()
        self.text_path = text_path
        self.img_path = img_path
        self.texti = TextIterator()
        if type == "train":
            self.text_data, self.img_data, self.id_list = self.texti.getTrainData()
        elif type == "test":
            self.text_data, self.img_data, self.id_list = self.texti.getTestData()
        else:
            self.text_data, self.img_data, self.id_list = self.texti.getValidData()


    def __getitem__(self, idx):
        img = self.img_data[idx]
        # text = self.text_data[idx]
        text = prompt_test.format(self.text_data[idx])
        label = self.texti.textData[self.id_list[idx]].label
        index = self.id_list[idx]
        return {'image':img, 'text':text, 'label':label,'index':index}
        # return img, text, label, index
    
    def __len__(self):
        return len(self.id_list)

class MSTIDataSet(Dataset):
    def __init__(self, type_path, tokenizer = None, max_examples=-1,
                 max_src_len=512, max_tgt_len=200):
        """
        max_examples: if > 0 then will load only max_examples into the dataset; -1 means use all

        max_src and max_tgt len refer to number of tokens in the input sequences
        # Note: these are not randomized. If they were we might need to collate.
        """
        self.max_examples = max_examples
        self.max_src_len = max_src_len  # max num of tokens in tokenize()
        self.max_tgt_len = max_tgt_len

        print('loading',type_path,'set')

        rationale_path = "{path_to_MSTI_rationale}.json"

        import json
        with open(rationale_path,'r') as f:
            self.rationale = json.load(f)

        self.type_path = type_path
        assert type_path in ['train','test','val']

        if tokenizer is None:
            self.tokenizer = T5TokenizerFast.from_pretrained('flan-t5-base')
        else:
            self.tokenizer  = tokenizer

        self._build()       # fill inputs, targets, max_lens
        print('done loading',type_path,'set with length of',len(self.img_data))

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):
        src_text = self.source_data[index]
        tgt_text = self.target_text[index]
        image_id = self.img_data[index]
        label = self.labels[index]

        img_id = self.id_list[index]
        box_path = os.path.join('./data/msti/Visual target labels3', img_id + '.txt')
        box_f = open(box_path, 'r', encoding='utf-8')
        bboxes = []
        box_labels = []

        for line in box_f.readlines():
            splited = line.strip().split()
            # print(splited)
            num_boxes = len(splited) // 5
            # print(num_boxes)
            for i in range(num_boxes):
                x1 = float(splited[0 + 5 * i])
                y1 = float(splited[1 + 5 * i])
                x2 = float(splited[2 + 5 * i])
                y2 = float(splited[3 + 5 * i])
                c = int(splited[4 + 5 * i])
                
                x, y, w, h = x1, y1, x2, y2

                bboxes.append(torch.tensor([x, y, w, h]))
                box_labels.append(torch.tensor(c))

        max_len = 10
        out_bboxes = np.array(bboxes, dtype=float)
        out_bboxes1 = np.zeros([max_len, 4])
        out_bboxes1[:min(out_bboxes.shape[0], max_len)] = out_bboxes[:min(out_bboxes.shape[0], max_len)]

        out_labels = np.array(box_labels, dtype=float)
        out_labels1 = np.zeros([10])
        out_labels1[:min(out_labels.shape[0], max_len)] = out_labels[:min(out_labels.shape[0], max_len)]
        
        return {"source_text": src_text,
                 "target_text": tgt_text,
                 "image_ids": image_id,
                 "bboxes": out_bboxes1,
                 "box_labels":out_labels1, 
                 "label": str(label)}  # for now
    
    def _build(self):

        print("buliding ",self.type_path," dataset...")

        source_data = []
        targets = [] 
        images = [] 
        dir = './data/msti/Textual target labels3'
        sentences, tgts, self.id_list,labels = load_sentence(dir,self.type_path)

        
        for text,target,idx in zip(sentences, tgts, self.id_list):
            rationale = self.rationale[idx]
            text = prompt_stage2.format(text,
                                rationale)
            images.append(f"./data/msti/img/{idx}.jpg")
            if target == []:
                target = 'None'
            else:
                target = ' '.join(target)
            targets.append(target)
            source_data.append(text)

        self.img_data = images
        self.source_data = source_data
        self.target_text = targets
        self.labels = labels
        # print(target)
        assert len(self.img_data) == len(targets)

def load_images(image_path):
    image_input_ids = []
    image_size = 600
    transform = Compose([
        RandomResize([image_size], max_size=image_size),
        Padding(max_x=image_size,max_y=image_size),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    for img_p in image_path:
        image = Image.open(img_p).convert("RGB") # load image
        # transform images
        image, _ = transform(image, None)
        image_input_ids.append(image)
    return image_input_ids

def load_images_with_source(image_path: list) -> Tuple[np.array, torch.Tensor]:
    image_input_ids = []    # keep a record of source image
    images = []

    transform = Compose(
        [
            RandomResize([600], max_size=600),
            Padding(max_x=600,max_y=600),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    
    for img_p in image_path:
        source_image = Image.open(img_p).convert("RGB") # load image
        image = np.asarray(source_image)
        image_transformed, _ = transform(source_image, None)
        image_input_ids.append(image_transformed)
        images.append(image)

    return images, image_input_ids

def load_image(image_path):
    # image_input_ids = []
    transform = Compose([
        Resize([800], max_size=1333),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB") # load image
    # transform images
    image, _ = transform(image, None)
    image_input_ids=image
    return image_input_ids


def bbox_transform(center_x, center_y, width, height):

    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2

    return x1,y1,x2,y2

def bbox_detransform(x1, y1, x2, y2):
    """
    xyxy to xywh
    """
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    width = x2 - x1
    height = y2 - y1

    return center_x, center_y, width, height


def get_dataloaders(args, tokenizer, batch_size, num_train = -1, num_val = -1, num_workers = 4, shuffle_train=True,
                    shuffle_dev=False) -> Tuple[DataLoader, DataLoader]:
    """
    Returns: Tuple[train_loader : DataLoader, dev_loader : DataLoader]
    # Note:
    # - we default to not shuffling the dev set

    """
    # todo: should pass max src and max tgt len in as arguments
    if args.stage == "stage1":
        if args.model_type == 't5':
            train_data_set = T5DataSet("train", tokenizer, max_examples=num_train,
                                    max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len)
            eval_data_set = T5DataSet("test", tokenizer, max_examples=num_val,
                                    max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len)
            test_data_set = T5DataSet("test", tokenizer, max_examples=num_val,
                                    max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len)
        else:
            train_data_set = RobertaDataSet("train", tokenizer, max_examples=num_train,
                                    max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len)
            eval_data_set = RobertaDataSet("val", tokenizer, max_examples=num_val,
                                    max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len)
            test_data_set = RobertaDataSet("test", tokenizer, max_examples=num_val,
                                    max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len)
    else:
        train_data_set = MSTIDataSet('train')
        eval_data_set = MSTIDataSet('val')
        test_data_set = MSTIDataSet('test')
    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_loader = DataLoader(eval_data_set, batch_size=batch_size, shuffle=shuffle_dev, num_workers=num_workers)
    test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=shuffle_dev, num_workers=num_workers)
    # log.info(f'Datasets loaded with sizes: train: {len(train_data_set)}, dev: {len(eval_data_set)}')

    return train_loader, eval_loader, test_loader


def forward(model, device, batch):
    tokenzier = model.tokenizer

    src_out = tokenzier(batch["source_text"], max_length=model.args.max_src_len, padding="max_length", return_tensors="pt", truncation=True)
    src_ids = src_out["input_ids"].to(device, dtype=torch.long)
    src_mask = src_out["attention_mask"].to(device, dtype=torch.long)

    if model.model_type == 't5':
        tgt_out = tokenzier(batch["target_text"], max_length=model.args.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True)
        tgt_ids = tgt_out["input_ids"].to(device, dtype=torch.long)
        tgt_ids[tgt_ids[: ,:] == 0 ] = -100
    else:
        tgt_ids = batch["target_text"].to(device, dtype=torch.float)

    if model.args.stage == "stage2":
        source_img, img_ids = load_images_with_source(batch["image_ids"])
        img_ids = torch.stack(img_ids).to(device, dtype=torch.float)
        tgt_boxes = batch['bboxes'].to(device, dtype=torch.float)

        for i,(boxes,src_img )in enumerate(zip(tgt_boxes,source_img)):
            h, w, _ = src_img.shape
            max_len = max(h,w)
            for j in range(len(boxes)):
                tgt_boxes[i][j] = tgt_boxes[i][j] / torch.tensor([max_len]*4).to(device, dtype=torch.float)
    else:
        img_ids = torch.stack(load_images(batch["image_ids"])).to(device, dtype=torch.float)

    label_ids = tgt_ids.to(device)
    out_dict,img_out = model.forward_train(image = img_ids, input_ids=src_ids, attention_mask=src_mask, labels=label_ids, return_dict=True,src_out = src_out)

    if model.args.stage == "stage2":
         # add bbox loss
        # resize pred bboxes back to original sizes for loss calculation
        pred_boxes = img_out['pred_boxes']#[:,:10,:]
        for i,(boxes,src_img )in enumerate(zip(pred_boxes,source_img)):
            h, w, _ = src_img.shape
            max_len = max(h,w)
            for j in range(len(boxes)):
                pred_boxes[i][j] = pred_boxes[i][j] * torch.tensor([max_len]*4).to(device, dtype=torch.float)
        img_out['pred_boxes'] = bbox_cxcywh_to_xyxy(pred_boxes)
        

        loss_cls, loss_bbox, loss_iou = \
        loss_by_feat_single(bbox_preds=img_out['pred_boxes'],
                    cls_scores=img_out['pred_logits'],
                    text_masks=src_mask,
                    box_labels=batch['box_labels'][:,:1],
                    target_boxes=batch['bboxes'][:,:1,:].cuda())       # we only take one target
        img_loss =  loss_cls*0.1 + loss_bbox*1e-3 + loss_iou*0.2       # to be adjusted

        out_dict['loss'] = out_dict['loss'] + img_loss
        # print(bbox_loss,out_dict['loss'].item())
        # assert False

    loss, logits = out_dict['loss'], out_dict['logits']

    if "source_expl" in batch.keys():
        src_out = tokenzier(batch["source_expl"], max_length=model.args.max_src_len, padding="max_length", return_tensors="pt", truncation=True)
        source_expl_ids = src_out["input_ids"].to(device, dtype=torch.long)
        src_mask = src_out["attention_mask"].to(device, dtype=torch.long)

        tgt_out = tokenzier(batch["target_expl"], max_length=model.args.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True)
        target_expl_ids = tgt_out["input_ids"].to(device, dtype=torch.long)

        target_expl_ids[target_expl_ids[: ,:] == 0 ] = -100
        target_expl_ids.to(device)
        out_dict,_ = model.forward_train(image = img_ids, input_ids=source_expl_ids, attention_mask=src_mask, labels=target_expl_ids, return_dict=True,src_out = src_out)
        loss = model.args.alpha * loss + (1. - model.args.alpha) * out_dict['loss']

    return loss, logits

def forward_test(model,device, batch):
    tokenzier = model.tokenizer
    
    src_out = tokenzier(batch["source_text"], max_length=model.args.max_src_len, padding="max_length", return_tensors="pt", truncation=True)
    src_ids = src_out["input_ids"].to(device, dtype=torch.long)
    src_mask = src_out["attention_mask"].to(device, dtype=torch.long)

    if model.model_type == 't5':
        tgt_out = tokenzier(batch["target_text"], max_length=model.args.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True)
        tgt_ids = tgt_out["input_ids"].to(device, dtype=torch.long)
        tgt_ids[tgt_ids[: ,:] == 0 ] = -100
    else:
        tgt_ids = batch["target_text"].to(device, dtype=torch.float)
    
    if model.args.stage == "stage2":
        source_img, img_ids = load_images_with_source(batch["image_ids"])
        img_ids = torch.stack(img_ids).to(device, dtype=torch.float)
        tgt_boxes = batch['bboxes'].to(device, dtype=torch.float)

        for i,(boxes,src_img )in enumerate(zip(tgt_boxes,source_img)):
            h, w, _ = src_img.shape
            max_len = max(h,w)
            for j in range(len(boxes)):
                tgt_boxes[i][j] = tgt_boxes[i][j] / torch.tensor([max_len]*4).to(device, dtype=torch.float)

        generated_ids,img_out = model(input_ids = src_ids,image = img_ids, attention_mask=src_mask,src_out = src_out) 

        # resize back to original sizes
        pred_boxes = img_out['pred_boxes']#[:,:10,:]
        for i,(boxes,src_img )in enumerate(zip(pred_boxes,source_img)):
            h, w, _ = src_img.shape
            max_len = max(h,w)
            for j in range(len(boxes)):
                pred_boxes[i][j] = pred_boxes[i][j] * torch.tensor([max_len]*4).to(device, dtype=torch.float)
        img_out['pred_boxes'] = bbox_cxcywh_to_xyxy(pred_boxes)         # model outputs xywh, xyxy needed for loss and val
        
        # if img_out['pred_logits'].shape[1]>1:
        #     img_out['pred_boxes'] = select_boxes(img_out)
        img_out['pred_boxes'] = img_out['pred_boxes'][:,:1,:]

    else:
        img_ids = torch.stack(load_images(batch["image_ids"])).to(device, dtype=torch.float)
        generated_ids,img_out = model(input_ids = src_ids,image = img_ids, attention_mask=src_mask,src_out = src_out) 

    return generated_ids,tgt_ids, img_out

def select_boxes(img_out):
    index = img_out['pred_logits'][:,:,0].argmax(dim=-1)
    pred_select = []
    for i,label in enumerate(index):
        t = img_out['pred_boxes'][i,label,:]
        pred_select.append(t)
    return torch.stack(pred_select).unsqueeze(dim=1)
