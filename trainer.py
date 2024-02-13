import torch
import torch.nn as nn

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup)
from models.util.slconfig import SLConfig
from utils import util

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import socket
from collections import OrderedDict
from pathlib import Path
from typing import *
from utils.loadData import *
from utils.eval import *

def trainer(model, args):
    log = util.get_logger(args.record_dir, "root", "debug")
    util.set_seed(args.seed)
    device, gpu_ids = util.get_available_devices()

    tokenizer = model.tokenizer
    # model.t5.parallelize()
    model.to(device)

    train_loader, dev_loader, test_loader = \
        get_dataloaders(args, tokenizer = tokenizer, batch_size=args.batch_size, num_train=args.num_train, num_val=args.num_val,
                        num_workers=args.num_workers)

    # reset in case we used the -1 flag for all
    num_train = len(train_loader.dataset)
    num_val = len(dev_loader.dataset)
    num_test = len(test_loader.dataset)
    total_steps = ( (num_train // args.batch_size) * args.epochs)     # num times that optim.step() will be called
    total_train = num_train * args.epochs
    print('data loaded, start training')


    # not needed
    # if args.stage == "stage1":
    #     for _, parameter in model.dino.named_parameters():
    #         parameter.requires_grad = False

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, eps=args.adam_eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_steps)
    save_path = args.save_path + args.exp_name +'.pth'

    log.info(f'device: {device}\n'
            f'gpu_ids: {gpu_ids}\n'
            f'total_steps: {total_steps}\n'
            f'total_train (num_t * epoch): {total_train}\n'
            f'learning rate: {args.lr}\n'
            f'save_dir: {save_path}\n'
            f'machine: {socket.gethostname()}\n')

    skip_training = args.skip_training
    if skip_training:
        print("Skip training...")
        args.epochs = 1

    epoch = 0       # number of times we have passed through entire set of training examples
    step = 0        # number of total examples we have done (will be epoch * len(data_set) at end of each epoch)
    best_acc = 0
    best_f1 = 0
    best_ap = 0
    while epoch < args.epochs:
        torch.cuda.empty_cache()
        if args.checkpoint is not None:
            log.info(f'Resume training with checkpoint {args.checkpoint}\n')
        epoch += 1
        model.train()
        if not skip_training:
            with torch.enable_grad(), tqdm(total=num_train) as progress_bar:
                for batch_num, batch in enumerate(train_loader):
                    if batch_num == 0:
                        log.info(f'**Training sample**\t Source: {batch["source_text"][0]}\t Target: {batch["target_text"][0]}\n')
                    batch_size = len(batch["image_ids"])
                    loss, _ = forward(model, device, batch)
                    loss_val = loss.item()      # get the item since loss is a tensor

                    # Backwardfor name, param in model.named_parameters()
                    for name, param in model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print("nan gradient found")
                            print("name:",name)
                            print("param:",param.grad)
                            raise SystemExit
                    # torch.autograd.set_detect_anomaly(True)
                    optimizer.zero_grad()
                    
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()        # don't need to pass step to scheduler

                    # Log info
                    step += batch_size
                    progress_bar.update(batch_size)
                    progress_bar.set_postfix(epoch=epoch,
                                            loss=loss_val)

        ###############
        # Evaluate (you might want to save checkpoints)
        ###############

        log.info(f'Evaluating at step {step}...')
        model.eval()        # put model in eval mode

        # See how the model is doing with exact match on tokens
        pred_list_all = []                      # accumulate for saving; list; one list per epoch
        loss_meter = util.AverageMeter()    # NLL (default metric for model) (reset each time)

        # set up two count variables
        total_matches_no_eos_ct = 0
        total_matches_with_eos_ct = 0
        last_epoch_loss = 100

        pred_list = []
        target_list = []
        acc = []
        EM = []
        box_res = []
        target_boxes = []

        with torch.no_grad(), \
            tqdm(total=num_val) as progress_bar:
            for batch_num, batch in enumerate(dev_loader):
                batch_size = len(batch["image_ids"])

                # evaluation for loss fcn
                loss, _ = forward(model, device, batch)     # loss, logits, but don't need logits
                loss_meter.update(loss.item(), batch_size)  # loss.item() since it's a tensor

                generated_ids,tgt_ids,img_out = forward_test(model,device,batch)
                orig_text_output = batch["target_text"]
                if args.model_type != 't5':
                    generated_ids = torch.tensor([[o] for o in generated_ids])
                    tgt_ids = torch.tensor([[i]for i in tgt_ids.argmax(dim=-1)])
                    orig_text_output = [int(i) for i in tgt_ids]

                # collect some stats
                total_matches_no_eos, total_matches_with_eos, _ = \
                    util.masked_token_match(tgt_ids, generated_ids, return_indices=True)
                total_matches_no_eos_ct += total_matches_no_eos
                total_matches_with_eos_ct += total_matches_with_eos

                # save for qualitative analysis
                # todo: this could break once skip_special_tokens is fixed
                if args.model_type == 't5':
                    outputs_decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                else:
                    outputs_decoded = [int(i) for i in generated_ids]
                text_id = [imgid[-22:-4] for imgid in batch['image_ids']]
                preds = list(zip(text_id, orig_text_output, outputs_decoded))

                if args.stage == "stage2":
                    pred, gts, batch_acc, batch_EM = batch_post_process(batch['label'], outputs_decoded)
                    EM.append(batch_EM)
                    pred_list.extend(pred)
                    target_list.extend(gts)
                    
                    for out, box in zip(img_out['pred_boxes'],batch['bboxes']):

                        mask = box > 0
                        length = int(len(box[mask])/4)
                        out = out[:length].cpu().detach().numpy()
                        box = box[:length].cpu().detach().numpy()
                        box_res.append(out)
                        target_boxes.append(box)
                        

                # print one batch of generations for qualitative assessment
                if batch_num == 0:
                    for orig_input, orig_target, actual_output in preds[:1]:
                        log.info(f'Source: {batch["source_text"][0]}\t Target: {orig_target}\n'
                                f'\t Actual: {actual_output}\n')
                    log.info(f'Target: {target_boxes}\n Actual: {box_res}\n')

                # Log info
                progress_bar.update(batch_size)
                progress_bar.set_postfix(NLL=loss_meter.avg)


            if args.stage == 'stage2':
                acc, f1, p, r, _ = evaluate_text(pred_list,target_list)
                EM = np.mean(EM)
                ap, ap50, ap75 = eval_ap_batch(box_res,target_boxes)
                log.info(f'Dev Acc: {acc}, F1: {f1}, EM: {EM}, Recall: {r}, Precision: {p}')
                log.info(f'Dev AP: {ap}, AP 50: {ap50}, AP 75: {ap75}\n')

            if loss_meter.avg > last_epoch_loss:
                epoch = args.epochs
            last_epoch_loss = loss_meter.avg
            
        results_list = [('NLL', loss_meter.avg),
                        ('exact_match_with_eos', total_matches_with_eos_ct),
                        ('exact_match_no_eos', total_matches_no_eos_ct)]
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        log.info(f'Dev {results_str}')

        ###############
        # Test (and to save results)
        ###############
        save = False
        if args.stage == 'stage1':
            if total_matches_with_eos_ct >= best_acc:
                best_acc = total_matches_with_eos_ct
                save = True
        else:
            if f1 > best_f1 or ap50 > best_ap:
                best_ap = max(best_ap,ap50)
                best_f1 = max(best_f1,f1)
                save = True
        if save == True:
            if not skip_training:
                if best_acc != 0:
                    log.info(f'New best matches {best_acc}...')
                else:
                    log.info(f'New best f1 {best_f1}, ap50 {best_ap}...')
                log.info(f'Saving model at epoch {epoch}...\n')
                model.save_state_dict(args)
            save = False
            
            pred_list_all = []
            pred_list = []
            target_list = []
            EM = []
            box_res = []
            target_boxes = []
            total_matches_no_eos_ct = 0
            total_matches_with_eos_ct = 0
            with torch.no_grad(), \
                tqdm(total=num_test) as progress_bar:
                for batch_num, batch in enumerate(test_loader):
                    batch_size = len(batch["image_ids"])

                    generated_ids,tgt_ids,img_out = forward_test(model,device,batch)
                    orig_text_output = batch["target_text"]
                    if args.model_type != 't5':
                        tgt_ids = tgt_ids.argmax(dim=-1)
                        orig_text_output = [int(i) for i in tgt_ids]

                    # collect some stats
                    if args.model_type == 't5':
                        total_matches_no_eos, total_matches_with_eos, _ = \
                            util.masked_token_match(tgt_ids, generated_ids, return_indices=True)
                    else:
                        total_matches_no_eos = int(sum(tgt_ids==generated_ids))
                        total_matches_with_eos = total_matches_no_eos
                    total_matches_no_eos_ct += total_matches_no_eos
                    total_matches_with_eos_ct += total_matches_with_eos

                    # todo: this could break once skip_special_tokens is fixed                    
                    if args.model_type == 't5':
                        outputs_decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    else:
                        outputs_decoded = [int(i) for i in generated_ids]
                    text_id = [imgid[-22:-4] for imgid in batch['image_ids']]
                    preds = list(zip(text_id, orig_text_output, outputs_decoded))
                    pred_list_all.extend(preds)

                    if args.stage == "stage2":
                        pred, gts , batch_acc, batch_EM = batch_post_process(batch['label'], outputs_decoded)
                        EM.append(batch_EM)
                        pred_list.extend(pred)
                        target_list.extend(gts)
                        for out, box in zip(img_out['pred_boxes'],batch['bboxes']):
                            mask = box > 0
                            length = int(len(box[mask])/4)
                            out = out[:length].cpu().detach().numpy()
                            box = box[:length].cpu().detach().numpy()
                            
                            box_res.append(out)
                            target_boxes.append(box)

                    # Log info
                    progress_bar.update(batch_size)
                    progress_bar.set_postfix(NLL=loss_meter.avg)
                    # save predictions for qualititative analysis

                if args.stage == 'stage2':
                    acc, f1, p, r, _ = evaluate_text(pred_list,target_list)
                    EM = np.mean(EM)
                    ap, ap50, ap75 = eval_ap_batch(box_res,target_boxes)
                    log.info(f'Test Acc: {acc}, F1: {f1}, EM: {EM}, Recall: {r}, Precision: {p}')
                    log.info(f'Test AP: {ap}, AP 50: {ap50}, AP 75: {ap75}\n')
                    util.save_csv_preds(pred_list_all, args.res_dir+'stage2')
                else:
                    util.save_csv_preds(pred_list_all, args.res_dir)

def test(model,args):
    log = util.get_logger(args.record_dir, "root", "debug")
    util.set_seed(args.seed)
    device, gpu_ids = util.get_available_devices()

    tokenizer = model.tokenizer
    # model.t5.parallelize()
    model.to(device)

    train_loader, dev_loader, test_loader = \
        get_dataloaders(args, tokenizer = tokenizer, batch_size=args.batch_size, num_train=args.num_train, num_val=args.num_val,
                        num_workers=args.num_workers)
    # test_loader = dev_loader    # for dev eval

    # reset in case we used the -1 flag for all
    num_train = len(train_loader.dataset)
    num_val = len(dev_loader.dataset)
    num_test = len(test_loader.dataset)
    total_steps = ( (num_train // args.batch_size) * args.epochs)     # num times that optim.step() will be called
    total_train = num_train * args.epochs
    print('data loaded, start testing')

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, eps=args.adam_eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_steps)
    save_path = args.save_path + args.exp_name +'.pth'

    log.info(f'device: {device}\n'
            f'gpu_ids: {gpu_ids}\n'
            f'total_steps: {total_steps}\n'
            f'total_train (num_t * epoch): {total_train}\n'
            f'learning rate: {args.lr}\n'
            f'model_dir: {args.checkpoint}\n'
            f'machine: {socket.gethostname()}\n')

    skip_training = args.skip_training
    if skip_training:
        print("Skip training...")
        args.epochs = 1

    epoch = 0       # number of times we have passed through entire set of training examples
    step = 0        # number of total examples we have done (will be epoch * len(data_set) at end of each epoch)
    best_acc = 0
    best_f1 = 0

    torch.cuda.empty_cache()

    log.info(f'Evaluating at step {step}...')
    model.eval()        # put model in eval mode

    # See how the model is doing with exact match on tokens
    pred_list_all = []                      # accumulate for saving; list; one list per epoch

    ###############
    # Test (and to save results)
    ###############    
    pred_list_all = []
    pred_list = []
    target_list = []
    EM = []
    box_res = []
    id_list = []
    target_boxes = []
    total_matches_no_eos_ct = 0
    total_matches_with_eos_ct = 0
    with torch.no_grad(), \
        tqdm(total=num_test) as progress_bar:
        for batch_num, batch in enumerate(test_loader):
            batch_size = len(batch["image_ids"])

            generated_ids,tgt_ids,img_out = forward_test(model,device,batch)
            orig_text_output = batch["target_text"]
            if args.model_type != 't5':
                tgt_ids = tgt_ids.argmax(dim=-1)
                orig_text_output = [int(i) for i in tgt_ids]

            # collect some stats
            if args.model_type == 't5':
                total_matches_no_eos, total_matches_with_eos, _ = \
                    util.masked_token_match(tgt_ids, generated_ids, return_indices=True)
            else:
                total_matches_no_eos = int(sum(tgt_ids==generated_ids))
                total_matches_with_eos = total_matches_no_eos
            total_matches_no_eos_ct += total_matches_no_eos
            total_matches_with_eos_ct += total_matches_with_eos

            # todo: this could break once skip_special_tokens is fixed                    
            if args.model_type == 't5':
                outputs_decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            else:
                outputs_decoded = [int(i) for i in generated_ids]
            text_id = [imgid[-22:-4] for imgid in batch['image_ids']]
            preds = list(zip(text_id, orig_text_output, outputs_decoded))
            pred_list_all.extend(preds)

            if args.stage == "stage2":
                pred, gts , batch_acc, batch_EM = batch_post_process(batch['label'], outputs_decoded)
                EM.append(batch_EM)
                ids = [ids[28:-4] for ids in batch["image_ids"]]
                id_list.extend(ids)
                pred_list.extend(pred)
                target_list.extend(gts)

                for out, box in zip(img_out['pred_boxes'],batch['bboxes']):
                    mask = box > 0
                    length = int(len(box[mask])/4)
                    out = out[:length].cpu().detach().numpy()
                    box = box[:length].cpu().detach().numpy()                  
                    box_res.append(out)
                    target_boxes.append(box)

            # Log info
            progress_bar.update(batch_size)
            # break
            # save predictions for qualititative analysis

        if args.stage == 'stage2':
            acc, f1, p, r, _ = evaluate_text(pred_list,target_list)
            EM = np.mean(EM)
            util.pred_to_csv(id_list,box_res,target_boxes)
            ap, ap50, ap75 = eval_ap_batch(box_res,target_boxes)
            log.info(f'Test Acc: {acc}, F1: {f1}, EM: {EM}, Recall: {r}, Precision: {p}')
            log.info(f'Test AP: {ap}, AP 50: {ap50}, AP 75: {ap75}\n')
        else:
            util.save_csv_preds(pred_list_all, args.res_dir)