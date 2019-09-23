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

def bind_model(model, optimizer=None):
    def load(filename, **kwargs):
        state = torch.load(os.path.join(filename, 'model.pt'))
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Model loaded')

    def save(filename, **kwargs):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(filename, 'model.pt'))

    def infer(wav_path):
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input = get_spectrogram_feature(wav_path).unsqueeze(0)
        input = input.to(device)

        logit = model(input_variable=input, input_lengths=None, teacher_forcing_ratio=0)
        logit = torch.stack(logit, dim=1).to(device)

        y_hat = logit.max(-1)[1]
        hyp = label_to_string(y_hat)

        return hyp[0]

    nsml.bind(save=save, load=load, infer=infer) # 'nsml.bind' function must be called at the end.


def main():

    global char2index
    global index2char
    global SOS_token
    global EOS_token
    global PAD_token

    parser = argparse.ArgumentParser(description='Speech hackathon lilililill model')
    parser.add_argument('--max_epochs', type=int, default=10, help='number of max epochs in training (default: 10)')
    parser.add_argument('--teacher_forcing', type=float, default=0.5, help='teacher forcing ratio in decoder (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--save_name', type=str, default='model', help='the name of model in nsml or local')
    parser.add_argument("--pause", type=int, default=0)

    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate in training (default: 0.2)')
    parser.add_argument('--lr', type=float, default=1e-03, help='learning rate (default: 0.001)')
    parser.add_argument('--num_mels', type=int, default=80, help='number of the mel bands (default: 80)')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training (default: 8)')
    parser.add_argument("--num_thread", type=int, default=2, help='number of the loading thread (default: 2)')
    parser.add_argument('--num_hidden_enc', type=int, default=1024, help='hidden size of model (default: 1024)')
    parser.add_argument('--num_hidden_dec', type=int, default=512, help='hidden size of model decoder (default: 512)')
    parser.add_argument('--nsc_in_ms', type=int, default=50, help='Number of sample size per time segment in ms (default: 50)')

    parser.add_argument('--memo', type=str, default='', help='Comment you wish to leave')
    parser.add_argument('--debug', type=str, default='False', help='debug mode')

    args = parser.parse_args()

    batch_size = args.batch_size
    num_thread = args.num_thread
    num_mels = args.num_mels

    char2index, index2char = load_label('./hackathon.labels')
    SOS_token = char2index['<s>']  # '<sos>' or '<s>'
    EOS_token = char2index['</s>']  # '<eos>' or '</s>'
    PAD_token = char2index['_']  # '-' or '_'

    unicode_jamo_list = My_Unicode_Jamo()
    logger.info(''.join(unicode_jamo_list))

    tokenizer = Tokenizer(unicode_jamo_list)
    jamo_tokens = tokenizer.word2num(unicode_jamo_list)
    logger.info('Tokens: {}'.format(jamo_tokens))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    net = Mel2SeqNet(num_mels, args.num_hidden_enc, args.num_hidden_dec, len(unicode_jamo_list), device)
    net_optimizer = optim.Adam(net.parameters(), lr=args.lr)
    ctc_loss = nn.CTCLoss().to(device)

    bind_model(net, net_optimizer)

    if args.pause == 1:
        nsml.paused(scope=locals())

    wav_paths, script_paths, korean_script_paths = get_paths(DATASET_PATH)
    logger.info('Korean script path 0: {}'.format(korean_script_paths[0]))

    if args.debug == 'True':
        idx = int(len(wav_paths)/50)
        wav_paths = wav_paths[:idx]
        script_paths = script_paths[:idx]
        korean_script_paths = korean_script_paths[:idx]

    logger.info('wav_paths len: {}'.format(len(wav_paths)))
    logger.info('script_paths len: {}'.format(len(script_paths)))
    logger.info('korean_script_paths len: {}'.format(len(korean_script_paths)))

    # Load Korean Scripts

    korean_script_list, jamo_script_list = get_korean_and_jamo_list(korean_script_paths)

    logger.info('Korean script 0: {}'.format(korean_script_list[0]))
    logger.info('Korean script 0 length: {}'.format(len(korean_script_list[0])))
    logger.info('Jamo script 0: {}'.format(jamo_script_list[0]))
    logger.info('Jamo script 0 length: {}'.format(len(jamo_script_list[0])))

    ground_truth_list = [(tokenizer.word2num(['<s>'] + list(jamo_script_list[i]) + ['</s>'])) for i in range(len(jamo_script_list))]

    # 90% of the data will be used as train
    split_index = int(0.9 * len(wav_paths))

    wav_path_list_train = wav_paths[:split_index]
    ground_truth_list_train = ground_truth_list[:split_index]
    korean_script_list_train = korean_script_list[:split_index]

    wav_path_list_eval = wav_paths[split_index:]
    ground_truth_list_eval = ground_truth_list[split_index:]
    korean_script_list_eval = korean_script_list[split_index:]

    logger.info('Total:Train:Eval = {}:{}:{}'.format(len(wav_paths), len(wav_path_list_train), len(wav_path_list_eval)))

    preloader_eval = Threading_Batched_Preloader(wav_path_list_eval, ground_truth_list_eval, korean_script_list_eval, batch_size, num_mels, args.nsc_in_ms)
    preloader_train = Threading_Batched_Preloader(wav_path_list_train, ground_truth_list_train, korean_script_list_train, batch_size, num_mels, args.nsc_in_ms)

    best_loss = 1e10

    # load all target scripts for reducing disk i/o
    target_path = os.path.join(DATASET_PATH, 'train_label')
    load_targets(target_path)

    logger.info('start')

    train_begin = time.time()

    for epoch in range(args.max_epochs):

        logger.info((datetime.now().strftime('%m-%d %H:%M:%S')))

        net.train()

        # preloader_train = Threading_Batched_Preloader(wav_path_list_train, ground_truth_list_train,
        #                                               korean_script_list_train, batch_size, num_mels)
        preloader_train.initialize_batch(num_thread)
        loss_list_train = list()

        logger.info("Initialized Training Preloader")
        count = 0
        total_dist = 0
        total_length = 0


        while not preloader_train.end_flag:
            batch = preloader_train.get_batch()
            # logger.info(psutil.virtual_memory())
            # logger.info("Got Batch")
            if batch is not None:
                # logger.info("Training Batch is not None")
                tensor_input, ground_truth, loss_mask, length_list, lev_ref_list = batch
                pred_tensor, loss = train(net, net_optimizer, ctc_loss, tensor_input.to(device),
                                          ground_truth.to(device), length_list.to(device), device)
                # logger.info(pred_tensor)
                loss_list_train.append(loss)

                lev_pred_list = Decode_Prediction(pred_tensor, tokenizer)
                dist, length = char_distance_list(lev_ref_list, lev_pred_list)
                total_dist += dist
                total_length += length

                count += 1

                if count % 10 == 0:
                    logger.info("Train {}/{}".format(count, int(np.ceil(len(wav_path_list_train) / batch_size))))
                    # logger.info("Train Loss {}".format(loss))
                    # logger.info("Train CER {}".format(dist / length))
                    logger.info("Shape of the Prediction Tensor: {}".format(pred_tensor.shape))

            else:
                logger.info("Training Batch is None")

        # del preloader_train

        # logger.info(loss_list_train)
        train_loss = np.mean(np.asarray(loss_list_train))
        train_cer = np.mean(np.asarray(total_dist/total_length))

        logger.info("Mean Train Loss: {}".format(train_loss))
        logger.info("Total Evaluation CER: {}".format(train_cer))

        # preloader_eval = Threading_Batched_Preloader(wav_path_list_eval, ground_truth_list_eval,
        #                                              korean_script_list_eval, batch_size, num_mels)
        preloader_eval.initialize_batch(num_thread)
        loss_list_eval = list()

        logger.info("Initialized Evaluation Preloader")

        count = 0
        total_dist = 0
        total_length = 0

        net.eval()

        while not preloader_eval.end_flag:
            batch = preloader_eval.get_batch()
            if batch is not None:
                tensor_input, ground_truth_, loss_mask, length_list, lev_ref_list = batch
                pred_tensor, loss = evaluate(net, ctc_loss, tensor_input.to(device), ground_truth_.to(device),
                                              length_list.to(device), device)
                loss_list_eval.append(loss)
                count += 1
                # if count % 5 == 0:
                logger.info("Eval {}/{}".format(count, int(np.ceil(len(wav_path_list_eval) / batch_size))))

                lev_pred_list = Decode_Prediction(pred_tensor, tokenizer)
                dist, length = char_distance_list(lev_ref_list, lev_pred_list)
                total_dist += dist
                total_length += length

                # logger.info("Eval Loss {}".format(loss))
                # logger.info("Eval CER {}".format(dist/length))

            else:
                logger.info("Training Batch is None")

        # del preloader_eval

        eval_cer = total_dist / total_length
        eval_loss = np.mean(np.asarray(loss_list_eval))
        logger.info("Mean Evaluation Loss: {}".format(eval_loss))
        logger.info("Total Evaluation CER: {}".format(eval_cer))

        nsml.report(False, step=epoch, train_epoch__loss=train_loss, train_epoch__cer=train_cer, eval__loss=eval_loss, eval__cer=eval_cer)

        best_model = (eval_loss < best_loss)
        nsml.save(args.save_name)

        if best_model:
            nsml.save('best')
            best_loss = eval_loss

if __name__ == "__main__":
    main()
