"""
Copyright 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#-*- coding: utf-8 -*-

import argparse
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
import Levenshtein as Lev
import jamotools

######
import psutil

from datetime import datetime
import scipy as sp
import numpy as np

from utils import *

import nsml
from nsml import GPU_NUM, DATASET_PATH, DATASET_NAME, HAS_DATASET

char2index = dict()
index2char = dict()
SOS_token = 0
EOS_token = 0
PAD_token = 0

if HAS_DATASET == False:
    DATASET_PATH = './sample_dataset'

DATASET_PATH = os.path.join(DATASET_PATH, 'train')

def label_to_string(labels):
    if len(labels.shape) == 1:
        sent = str()
        for i in labels:
            if i.item() == EOS_token:
                break
            sent += index2char[i.item()]
        return sent

    elif len(labels.shape) == 2:
        sents = list()
        for i in labels:
            sent = str()
            for j in i:
                if j.item() == EOS_token:
                    break
                sent += index2char[j.item()]
            sents.append(sent)

        return sents


def char_distance(ref, hyp):
    ref = ref.replace(' ', '') 
    hyp = hyp.replace(' ', '') 

    dist = Lev.distance(hyp, ref)
    length = len(ref.replace(' ', ''))

    return dist, length 


def get_distance(ref_labels, hyp_labels, display=False):
    total_dist = 0
    total_length = 0
    for i in range(len(ref_labels)):
        ref = label_to_string(ref_labels[i])
        hyp = label_to_string(hyp_labels[i])
        dist, length = char_distance(ref, hyp)
        total_dist += dist
        total_length += length 
        if display:
            cer = total_dist / total_length
            logger.debug('%d (%0.4f)\n(%s)\n(%s)' % (i, cer, ref, hyp))
    return total_dist, total_length


def bind_model(net, net_B, net_optimizer=None, net_B_optimizer=None, index2char=None, tokenizer=None):
    def load(filename, **kwargs):
        state = torch.load(os.path.join(filename, 'modelA.pt'))
        net.load_state_dict(state['model'])
        if 'optimizer' in state and net_optimizer:
            net_optimizer.load_state_dict(state['optimizer'])

        state = torch.load(os.path.join(filename, 'modelB.pt'))
        net_B.load_state_dict(state['model'])
        if 'optimizer' in state and net_B_optimizer:
            net_B_optimizer.load_state_dict(state['optimizer'])

        print('Model loaded')

    def save(filename, **kwargs):
        state = {
            'model': net.state_dict(),
            'optimizer': net_optimizer.state_dict()
        }
        torch.save(state, os.path.join(filename, 'modelA.pt'))

        state = {
            'model': net_B.state_dict(),
            'optimizer': net_B_optimizer.state_dict()
        }
        torch.save(state, os.path.join(filename, 'modelB.pt'))

    def infer(wav_path):
        net.eval()
        net_B.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input = CREATE_MEL(wav_path, 40)
        input = input.type(torch.FloatTensor).to(device)

        pred_tensor = net(input)

        lev_input = Decode_CTC_Prediction_And_Batch(pred_tensor)
        lev_pred = net_B.net_infer(lev_input.to(device))
        pred_string_list = Decode_Lev_Prediction(lev_pred, index2char)
        answer = pred_string_list[0].replace(' ', '')

        # jamo_result = Decode_Prediction_No_Filtering(pred_tensor, tokenizer)
        # answer = (jamotools.join_jamos(jamo_result[0]).replace('<s>', ''))

        return answer

    nsml.bind(save=save, load=load, infer=infer) # 'nsml.bind' function must be called at the end.


def main():

    global char2index
    global index2char
    global SOS_token
    global EOS_token
    global PAD_token

    parser = argparse.ArgumentParser(description='Speech hackathon lilililill model')
    parser.add_argument('--max_epochs', type=int, default=100, help='number of max epochs in training (default: 100)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--save_name', type=str, default='model', help='the name of model in nsml or local')

    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate in training (default: 0.2)')
    parser.add_argument('--lr', type=float, default=1e-03, help='learning rate (default: 0.001)')
    parser.add_argument('--num_mels', type=int, default=80, help='number of the mel bands (default: 80)')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size in training (default: 128)')
    parser.add_argument("--num_thread", type=int, default=4, help='number of the loading thread (default: 4)')
    parser.add_argument('--num_hidden_enc', type=int, default=1024, help='hidden size of model (default: 1024)')
    parser.add_argument('--num_hidden_dec', type=int, default=512, help='hidden size of model decoder (default: 512)')
    parser.add_argument('--nsc_in_ms', type=int, default=50, help='Number of sample size per time segment in ms (default: 50)')

    parser.add_argument('--ref_repeat', type=int, default=1, help='Number of repetition of reference seq2seq (default: 1)')
    parser.add_argument('--loss_lim', type=float, default=0.05, help='Minimum loss threshold (default: 0.05)')

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument("--pause", type=int, default=0)
    parser.add_argument('--memo', type=str, default='', help='Comment you wish to leave')
    parser.add_argument('--debug', type=str, default='False', help='debug mode')

    parser.add_argument('--load', type=str, default=None)

    args = parser.parse_args()

    batch_size = args.batch_size
    num_thread = args.num_thread
    num_mels = args.num_mels

    char2index, index2char = load_label('./hackathon.labels')
    SOS_token = char2index['<s>']  # '<sos>' or '<s>'
    EOS_token = char2index['</s>']  # '<eos>' or '</s>'
    PAD_token = char2index['_']  # '-' or '_'

    unicode_jamo_list = My_Unicode_Jamo_v2()
    # logger.info(''.join(unicode_jamo_list))

    # logger.info('This is a new main2.py')

    tokenizer = Tokenizer(unicode_jamo_list)
    jamo_tokens = tokenizer.word2num(unicode_jamo_list)
    # logger.info('Tokens: {}'.format(jamo_tokens))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    net = Mel2SeqNet_v2(num_mels, args.num_hidden_enc, args.num_hidden_dec, len(unicode_jamo_list), device)
    net_optimizer = optim.Adam(net.parameters(), lr=args.lr)
    ctc_loss = nn.CTCLoss().to(device)

    # net_B = Seq2SeqNet(512, jamo_tokens, char2index, device) #########
    net_B = Seq2SeqNet_v2(1024, jamo_tokens, char2index, device)  #########
    net_B_optimizer = optim.Adam(net_B.parameters(), lr=args.lr) #########
    net_B_criterion = nn.NLLLoss(reduction='none').to(device) #########

    bind_model(net, net_B, net_optimizer, net_B_optimizer, index2char, tokenizer)

    if args.pause == 1:
        nsml.paused(scope=locals())

    if args.mode != "train":
        return

    if args.load != None:
        nsml.load(checkpoint='best', session='team47/sr-hack-2019-dataset/' + args.load)
        nsml.save('saved')

    for g in net_optimizer.param_groups:
        g['lr'] = 1e-05

    for g in net_B_optimizer.param_groups:
        g['lr'] = 1e-05

    for g in net_optimizer.param_groups:
        logger.info(g['lr'])

    for g in net_B_optimizer.param_groups:
        logger.info(g['lr'])

    wav_paths, script_paths, korean_script_paths = get_paths(DATASET_PATH)
    logger.info('Korean script path 0: {}'.format(korean_script_paths[0]))

    logger.info('wav_paths len: {}'.format(len(wav_paths)))
    logger.info('script_paths len: {}'.format(len(script_paths)))
    logger.info('korean_script_paths len: {}'.format(len(korean_script_paths)))

    # Load Korean Scripts

    korean_script_list, jamo_script_list = get_korean_and_jamo_list_v2(korean_script_paths)

    logger.info('Korean script 0: {}'.format(korean_script_list[0]))
    logger.info('Korean script 0 length: {}'.format(len(korean_script_list[0])))
    logger.info('Jamo script 0: {}'.format(jamo_script_list[0]))
    logger.info('Jamo script 0 length: {}'.format(len(jamo_script_list[0])))

    script_path_list = get_script_list(script_paths, SOS_token, EOS_token)

    ground_truth_list = [(tokenizer.word2num(['<s>'] + list(jamo_script_list[i]) + ['</s>'])) for i in range(len(jamo_script_list))]

    # 90% of the data will be used as train
    # split_index = int(0.9 * len(wav_paths))
    split_index = int(0.95 * len(wav_paths))

    wav_path_list_train = wav_paths[:split_index]
    ground_truth_list_train = ground_truth_list[:split_index]
    korean_script_list_train = korean_script_list[:split_index]
    script_path_list_train = script_path_list[:split_index]

    wav_path_list_eval = wav_paths[split_index:]
    ground_truth_list_eval = ground_truth_list[split_index:]
    korean_script_list_eval = korean_script_list[split_index:]
    script_path_list_eval = script_path_list[split_index:]

    logger.info('Total:Train:Eval = {}:{}:{}'.format(len(wav_paths), len(wav_path_list_train), len(wav_path_list_eval)))

    preloader_eval = Threading_Batched_Preloader_v2(wav_path_list_eval, ground_truth_list_eval, script_path_list_eval, korean_script_list_eval, batch_size, num_mels, args.nsc_in_ms, is_train=True)
    preloader_train = Threading_Batched_Preloader_v2(wav_path_list_train, ground_truth_list_train, script_path_list_train, korean_script_list_train, batch_size, num_mels, args.nsc_in_ms, is_train=False)

    best_loss = 1e10
    best_eval_cer = 1e10

    # load all target scripts for reducing disk i/o
    target_path = os.path.join(DATASET_PATH, 'train_label')
    load_targets(target_path)

    logger.info('start')

    train_begin = time.time()

    for epoch in range(args.max_epochs):

        logger.info((datetime.now().strftime('%m-%d %H:%M:%S')))

        net.train()
        net_B.train()

        preloader_train.initialize_batch(num_thread)
        loss_list_train = list()
        seq2seq_loss_list_train = list()
        seq2seq_loss_list_train_ref = list()

        logger.info("Initialized Training Preloader")
        count = 0

        total_dist = 0
        total_length = 1
        total_dist_ref = 0
        total_length_ref = 1

        while not preloader_train.end_flag:
            batch = preloader_train.get_batch()
            # logger.info(psutil.virtual_memory())
            # logger.info("Got Batch")
            if batch is not None:
                # logger.info("Training Batch is not None")
                tensor_input, ground_truth, loss_mask, length_list, batched_num_script, batched_num_script_loss_mask = batch
                pred_tensor, loss = train(net, net_optimizer, ctc_loss, tensor_input.to(device),
                                          ground_truth.to(device), length_list.to(device), device)
                loss_list_train.append(loss)

                ####################################################

                jamo_result = Decode_Prediction_No_Filtering(pred_tensor, tokenizer)

                true_string_list = Decode_Num_Script(batched_num_script.detach().cpu().numpy(), index2char)

                for i in range(args.ref_repeat):
                    lev_input_ref = ground_truth

                    lev_pred_ref, attentions_ref, seq2seq_loss_ref = net_B.net_train(lev_input_ref.to(device),
                                                                                             batched_num_script.to(device),
                                                                                             batched_num_script_loss_mask.to(device),
                                                                                             net_B_optimizer,
                                                                                             net_B_criterion)

                pred_string_list_ref = Decode_Lev_Prediction(lev_pred_ref, index2char)
                seq2seq_loss_list_train_ref.append(seq2seq_loss_ref)
                dist_ref, length_ref = char_distance_list(true_string_list, pred_string_list_ref)

                pred_string_list = [None]

                dist = 0
                length = 0

                if (loss < args.loss_lim):
                    lev_input = Decode_CTC_Prediction_And_Batch(pred_tensor)
                    lev_pred, attentions, seq2seq_loss = net_B.net_train(lev_input.to(device),
                                                                                 batched_num_script.to(device),
                                                                                 batched_num_script_loss_mask.to(device),
                                                                                 net_B_optimizer, net_B_criterion)
                    pred_string_list = Decode_Lev_Prediction(lev_pred, index2char)
                    seq2seq_loss_list_train.append(seq2seq_loss)
                    dist, length = char_distance_list(true_string_list, pred_string_list)

                total_dist_ref += dist_ref
                total_length_ref += length_ref

                total_dist += dist
                total_length += length

                count += 1

                if count % 25 == 0:
                    logger.info("Train: Count {} | {} => {}".format(count, true_string_list[0], pred_string_list_ref[0]))

                    logger.info("Train: Count {} | {} => {} => {}".format(count, true_string_list[0], jamo_result[0],
                                                             pred_string_list[0]))

            else:
                logger.info("Training Batch is None")

        # del preloader_train

        # logger.info(loss_list_train)
        train_loss = np.mean(np.asarray(loss_list_train))
        train_cer = np.mean(np.asarray(total_dist/total_length))
        train_cer_ref = np.mean(np.asarray(total_dist_ref / total_length_ref))

        logger.info("Mean Train Loss: {}".format(train_loss))
        logger.info("Total Train CER: {}".format(train_cer))
        logger.info("Total Train Reference CER: {}".format(train_cer_ref))

        preloader_eval.initialize_batch(num_thread)
        loss_list_eval = list()
        seq2seq_loss_list_eval = list()
        seq2seq_loss_list_eval_ref = list()

        logger.info("Initialized Evaluation Preloader")

        count = 0
        total_dist = 0
        total_length = 1
        total_dist_ref = 0
        total_length_ref = 1

        net.eval()
        net_B.eval()

        while not preloader_eval.end_flag:
            batch = preloader_eval.get_batch()
            if batch is not None:
                tensor_input, ground_truth, loss_mask, length_list, batched_num_script, batched_num_script_loss_mask = batch
                pred_tensor, loss = evaluate(net, ctc_loss, tensor_input.to(device), ground_truth.to(device), length_list.to(device), device)
                loss_list_eval.append(loss)

                ####################

                jamo_result = Decode_Prediction_No_Filtering(pred_tensor, tokenizer)

                true_string_list = Decode_Num_Script(batched_num_script.detach().cpu().numpy(), index2char)

                lev_input_ref = ground_truth
                lev_pred_ref, attentions_ref, seq2seq_loss_ref = net_B.net_eval(lev_input_ref.to(device),
                                                                                 batched_num_script.to(device),
                                                                                 batched_num_script_loss_mask.to(device),
                                                                                 net_B_criterion)

                pred_string_list_ref = Decode_Lev_Prediction(lev_pred_ref, index2char)
                seq2seq_loss_list_train_ref.append(seq2seq_loss_ref)
                dist_ref, length_ref = char_distance_list(true_string_list, pred_string_list_ref)

                lev_input = Decode_CTC_Prediction_And_Batch(pred_tensor)
                lev_pred, attentions, seq2seq_loss = net_B.net_eval(lev_input.to(device),
                                                                     batched_num_script.to(device),
                                                                     batched_num_script_loss_mask.to(device),
                                                                     net_B_criterion)
                pred_string_list = Decode_Lev_Prediction(lev_pred, index2char)
                seq2seq_loss_list_train.append(seq2seq_loss)
                dist, length = char_distance_list(true_string_list, pred_string_list)

                total_dist_ref += dist_ref
                total_length_ref += length_ref

                total_dist += dist
                total_length += length

                count += 1

                ####################

                if count % 5 == 0:
                    logger.info("Eval: Count {} | {} => {}".format(count, true_string_list[0], pred_string_list_ref[0]))

                    logger.info("Eval: Count {} | {} => {} => {}".format(count, true_string_list[0], jamo_result[0],
                                                             pred_string_list[0]))

            else:
                logger.info("Training Batch is None")

        eval_cer = total_dist / total_length
        eval_cer_ref = total_dist_ref / total_length_ref
        eval_loss = np.mean(np.asarray(loss_list_eval))

        logger.info("Mean Evaluation Loss: {}".format(eval_loss))
        logger.info("Total Evaluation CER: {}".format(eval_cer))
        logger.info("Total Evaluation Reference CER: {}".format(eval_cer_ref))

        nsml.report(False, step=epoch, train_epoch__loss=train_loss, train_epoch__cer=train_cer,
                    train_epoch__cer_ref=train_cer_ref,
                    eval__loss=eval_loss, eval__cer=eval_cer, eval__cer_ref=eval_cer_ref)

        nsml.save(args.save_name)
        best_model = (eval_cer < best_eval_cer)
        if best_model:
            nsml.save('best')
            best_eval_cer = eval_cer

        logger.info("Inference Check")

        net.eval()
        net_B.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for wav_path in wav_path_list_eval:
            input = CREATE_MEL(wav_path, 40)
            input = input.type(torch.FloatTensor).to(device)

            pred_tensor = net(input)

            jamo_result = Decode_Prediction_No_Filtering(pred_tensor, tokenizer)

            lev_input = Decode_CTC_Prediction_And_Batch(pred_tensor)
            lev_pred = net_B.net_infer(lev_input.to(device))
            pred_string_list = Decode_Lev_Prediction(lev_pred, index2char)

            logger.info(pred_string_list[0])
            logger.info(jamotools.join_jamos(jamo_result[0]).replace('<s>', ''))

if __name__ == "__main__":
    main()
