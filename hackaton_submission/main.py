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

import os
import sys
import time
import math
import wavio
import argparse
import queue
import shutil
import random
import math
import time
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
import Levenshtein as Lev

import jamotools
import copy
from datetime import datetime
import scipy as sp
import numpy as np
import librosa
import re

import label_loader
from loader import *
from models import EncoderRNN, DecoderRNN, Seq2seq

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

#
# def train(model, total_batch_size, queue, criterion, optimizer, device, train_begin, train_loader_count, print_batch=5, teacher_forcing_ratio=1):
#     total_loss = 0.
#     total_num = 0
#     total_dist = 0
#     total_length = 0
#     total_sent_num = 0
#     batch = 0
#
#     model.train()
#
#     logger.info('train() start')
#
#     begin = epoch_begin = time.time()
#
#     while True:
#         if queue.empty():
#             logger.debug('queue is empty')
#
#         feats, scripts, feat_lengths, script_lengths = queue.get()
#
#         logger.info('Shape of the feats: {}'.format(feats.shape))
#
#         if feats.shape[0] == 0:
#             # empty feats means closing one loader
#             train_loader_count -= 1
#
#             logger.debug('left train_loader: %d' % (train_loader_count))
#
#             if train_loader_count == 0:
#                 break
#             else:
#                 continue
#
#         optimizer.zero_grad()
#
#         logger.info('Shape of the feats: {}'.format(feats.shape))
#
#         feats = feats.to(device)
#         scripts = scripts.to(device)
#
#         src_len = scripts.size(1)
#         target = scripts[:, 1:]
#
#         model.module.flatten_parameters()
#         logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=teacher_forcing_ratio)
#
#         logit = torch.stack(logit, dim=1).to(device)
#
#         y_hat = logit.max(-1)[1]
#
#         loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
#         total_loss += loss.item()
#         total_num += sum(feat_lengths)
#
#         display = random.randrange(0, 100) == 0
#         dist, length = get_distance(target, y_hat, display=display)
#         total_dist += dist
#         total_length += length
#
#         total_sent_num += target.size(0)
#
#         loss.backward()
#         optimizer.step()
#
#         if batch % print_batch == 0:
#             current = time.time()
#             elapsed = current - begin
#             epoch_elapsed = (current - epoch_begin) / 60.0
#             train_elapsed = (current - train_begin) / 3600.0
#
#             logger.info('batch: {:4d}/{:4d}, loss: {:.4f}, cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h'
#                 .format(batch,
#                         #len(dataloader),
#                         total_batch_size,
#                         total_loss / total_num,
#                         total_dist / total_length,
#                         elapsed, epoch_elapsed, train_elapsed))
#             begin = time.time()
#
#             nsml.report(False,
#                         step=train.cumulative_batch_count, train_step__loss=total_loss/total_num,
#                         train_step__cer=total_dist/total_length)
#         batch += 1
#         train.cumulative_batch_count += 1
#
#     logger.info('train() completed')
#     return total_loss / total_num, total_dist / total_length
#
#
# train.cumulative_batch_count = 0
#
#
# def evaluate(model, dataloader, queue, criterion, device):
#     logger.info('evaluate() start')
#     total_loss = 0.
#     total_num = 0
#     total_dist = 0
#     total_length = 0
#     total_sent_num = 0
#
#     model.eval()
#
#     with torch.no_grad():
#         while True:
#             feats, scripts, feat_lengths, script_lengths = queue.get()
#             if feats.shape[0] == 0:
#                 break
#
#             feats = feats.to(device)
#             scripts = scripts.to(device)
#
#             src_len = scripts.size(1)
#             target = scripts[:, 1:]
#
#             model.module.flatten_parameters()
#             logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=0.0)
#
#             logit = torch.stack(logit, dim=1).to(device)
#             y_hat = logit.max(-1)[1]
#
#             loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
#             total_loss += loss.item()
#             total_num += sum(feat_lengths)
#
#             display = random.randrange(0, 100) == 0
#             dist, length = get_distance(target, y_hat, display=display)
#             total_dist += dist
#             total_length += length
#             total_sent_num += target.size(0)
#
#     logger.info('evaluate() completed')
#     return total_loss / total_num, total_dist / total_length

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

def split_dataset(config, wav_paths, script_paths, korean_script_paths, valid_ratio=0.05):
    train_loader_count = config.workers
    records_num = len(wav_paths)
    batch_num = math.ceil(records_num / config.batch_size)

    valid_batch_num = math.ceil(batch_num * valid_ratio)
    train_batch_num = batch_num - valid_batch_num

    batch_num_per_train_loader = math.ceil(train_batch_num / config.workers)

    train_begin = 0
    train_end_raw_id = 0
    train_dataset_list = list()

    for i in range(config.workers):

        train_end = min(train_begin + batch_num_per_train_loader, train_batch_num)

        train_begin_raw_id = train_begin * config.batch_size
        train_end_raw_id = train_end * config.batch_size

        train_dataset_list.append(BaseDataset(
                                        wav_paths[train_begin_raw_id:train_end_raw_id],
                                        script_paths[train_begin_raw_id:train_end_raw_id],
                                        korean_script_paths[train_begin_raw_id:train_end_raw_id],
                                        SOS_token, EOS_token))
        train_begin = train_end 

    valid_dataset = BaseDataset(wav_paths[train_end_raw_id:], script_paths[train_end_raw_id:], korean_script_paths[train_end_raw_id:],SOS_token, EOS_token)

    return train_batch_num, train_dataset_list, valid_dataset


# My Tokenizer
class Tokenizer():
    def __init__(self, vocabs):
        self.vocabs = vocabs

    def word2num(self, sentence):
        tokens = list()
        for char in sentence:
            tokens.append(self.vocabs.index(char))
        return tokens

    def word2vec(self, sentence):
        vectors = np.zeros((len(sentence), len(self.vocabs)))
        for i, char in enumerate(sentence):
            vectors[i, self.vocabs.index(char)] = 1
        return vectors

    def num2word(self, num):
        output = list()
        for i in num:
            output.append(self.vocabs[i])
        return output

    def num2vec(self, numbers):
        vectors = np.zeros((len(numbers), len(self.vocabs)))
        for i, num in enumerate(numbers):
            vectors[i, num] = 1
        return vectors

# My Encoder
class Encoder(nn.Module):
    def __init__(self, D_in, H):
        super(Encoder, self).__init__()
        self.fc = torch.nn.Linear(D_in, H)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.gru = nn.GRU(H, int(H / 2), bidirectional=True, batch_first=True)

    def forward(self, input_tensor):
        # (B, T, F)
        output_tensor = self.fc(input_tensor)
        output_tensor = self.relu(output_tensor)
        output_tensor = self.dropout(output_tensor)
        # (B, T, H)
        output_tensor, _ = self.gru(output_tensor)
        return output_tensor

# My Decoder
class CTC_Decoder(nn.Module):
    def __init__(self, H, D_out, num_chars):
        super(CTC_Decoder, self).__init__()
        self.fc_embed = nn.Linear(H, H)
        self.relu_embed = torch.nn.ReLU()
        self.dropout_embed = nn.Dropout(p=0.5)
        self.gru = nn.GRU(H, D_out, batch_first=True)
        self.fc = nn.Linear(D_out, num_chars)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, input_tensor):
        # (B, T, 2 * H/2)
        output_tensor = self.fc_embed(input_tensor)
        output_tensor = self.relu_embed(output_tensor)
        output_tensor = self.dropout_embed(output_tensor)
        # (B, T, H)
        output_tensor, _ = self.gru(input_tensor)
        # (B, T, H)
        output_tensor = self.fc(output_tensor)
        # (B, T, 75)
        prediction_tensor = self.log_softmax(output_tensor)

        return prediction_tensor

class Mel2SeqNet(nn.Module):
    def __init__(self, D_in, H, D_out, num_chars, device):
        super(Mel2SeqNet, self).__init__()

        self.encoder = Encoder(D_in, H).to(device)
        self.decoder = CTC_Decoder(H, D_out, num_chars).to(device)

        # Initialize weights with random uniform numbers with range
        for param in self.encoder.parameters():
            param.data.uniform_(-0.1, 0.1)
        for param in self.decoder.parameters():
            param.data.uniform_(-0.1, 0.1)

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        # (B, T, F) -> (B, T, H)
        encoded_tensor = self.encoder(input_tensor)
        # (B, T, H) -> (B, T, 75)
        pred_tensor = self.decoder(encoded_tensor)
        pred_tensor = pred_tensor.permute(1, 0, 2)

        return pred_tensor


class Threading_Batched_Preloader():
    def __init__(self, wav_path_list, ground_truth_list, batch_size):
        super(Threading_Batched_Preloader).__init__()
        self.wav_path_list = wav_path_list
        self.total_num_input = len(wav_path_list)
        self.tensor_input_list = [None] * self.total_num_input
        self.ground_truth_list = ground_truth_list
        self.sentence_length_list = np.asarray(list(map(len, ground_truth_list)))
        self.shuffle_step = 12
        self.loading_sequence = None
        self.end_flag = False
        self.batch_size = batch_size
        self.queue = queue.Queue(32)
        self.thread_flags = list()

    # Shuffle loading index and set end flag to false
    def initialize_batch(self, thread_num):
        loading_sequence = np.argsort(self.sentence_length_list)
        bundle = np.stack([self.sentence_length_list[loading_sequence], loading_sequence])

        for seq_len in range(self.shuffle_step, np.max(self.sentence_length_list), self.shuffle_step):
            idxs = np.where((bundle[0, :] > seq_len) & (bundle[0, :] <= seq_len + self.shuffle_step))[0]
            idxs_origin = copy.deepcopy(idxs)
            random.shuffle(idxs)
            bundle[:, idxs_origin] = bundle[:, idxs]

        loading_sequence = bundle[1, :]
        loading_sequence_len = len(loading_sequence)

        thread_size = int(np.ceil(loading_sequence_len / thread_num))

        load_idxs_list = list()
        for i in range(thread_num):
            start_idx = i * thread_size
            end_idx = (i + 1) * thread_size

            if end_idx > loading_sequence_len:
                end_idx = loading_sequence_len

            load_idxs_list.append(loading_sequence[start_idx:end_idx])

        self.end_flag = False

        self.queue = queue.Queue(32)
        self.thread_flags = [False] * thread_num

        self.thread_list = [
            Batching_Thread(self.wav_path_list, self.ground_truth_list, load_idxs_list[i], self.queue, self.batch_size,
                            self.thread_flags, i) for i in range(thread_num)]

        for thread in self.thread_list:
            thread.start()
        return

    def check_thread_flags(self):
        for flag in self.thread_flags:
            if flag == False:
                return False

        if (self.queue.empty):
            self.end_flag = True
            return True

        return False

    def get_batch(self):
        # logger.info("Called get_batch ")
        while not (self.check_thread_flags()):
            batch = self.queue.get()

            if (batch != None):
                # logger.info("Batch is Ready")
                batched_tensor = batch[0]
                batched_ground_truth = batch[1]
                batched_loss_mask = batch[2]
                ground_truth_size_list = batch[3]

                return batched_tensor, batched_ground_truth, batched_loss_mask, ground_truth_size_list

            else:
                logger.info("Loader Queue is empty")
                time.sleep(1)

        return None


class Batching_Thread(threading.Thread):

    def __init__(self, wav_path_list, ground_truth_list, load_idxs_list, queue, batch_size, thread_flags, id):

        threading.Thread.__init__(self)
        self.wav_path_list = wav_path_list
        self.ground_truth_list = ground_truth_list
        self.load_idxs_list = load_idxs_list
        self.list_len = len(load_idxs_list)
        self.cur_idx = 0
        self.id = id
        self.queue = queue
        self.batch_size = batch_size
        self.thread_flags = thread_flags

        logger.info("Batching Thread {} Initialized".format(self.id))

    def run(self):

        while (self.cur_idx < self.list_len):
            batch = self.batch()
            success = False
            while success == False:
                try:
                    self.queue.put(batch, True, 4)
                    success = True
                except:
                    # logger.info("Batching Failed in Thread ID: {} Queue Size: {}".format(self.id, self.queue.qsize()))
                    time.sleep(1)

        self.thread_flags[self.id] = True

        #         print("Thread {} finished".foramt(self.id))

        return

    def batch(self):

        tensor_list = list()
        ground_truth_list = list()
        tensor_size_list = list()
        ground_truth_size_list = list()

        count = 0
        max_seq_len = 0
        max_sen_len = 0

        for i in range(self.batch_size):

            # If there is no more file, break and set end_flag true
            if self.cur_idx >= self.list_len:
                self.end_flag = True
                break

            wav_path = self.wav_path_list[self.load_idxs_list[self.cur_idx]]

            tensor = self.create_mel(wav_path)
            tensor_list.append(tensor)
            tensor_size_list.append(tensor.shape[1])

            ground_truth = self.ground_truth_list[self.load_idxs_list[self.cur_idx]]
            ground_truth_list.append(ground_truth)
            ground_truth_size_list.append(len(ground_truth))

            if (tensor.shape[1] > max_seq_len):
                max_seq_len = tensor.shape[1]
            if (len(ground_truth) > max_sen_len):
                max_sen_len = len(ground_truth)

            self.cur_idx += 1
            count += 1

        batched_tensor = torch.zeros(count, max_seq_len + 5, n_mels)
        batched_ground_truth = torch.zeros(count, max_sen_len)
        batched_loss_mask = torch.zeros(count, max_sen_len)
        ground_truth_size_list = torch.tensor(np.asarray(ground_truth_size_list), dtype=torch.long)

        for order in range(count):

            target = tensor_list[order]

            pad_random = np.random.randint(0, 5)

            # Time shift, add zeros in front of an image
            if pad_random > 0:
                offset = torch.zeros(target.shape[0], pad_random, target.shape[2], dtype=torch.float64)
                target = torch.cat((offset, target), 1)

            # Add random noise
            target = target + (torch.rand(target.shape, dtype=torch.float64) - 0.5) / 20

            # Value less than 0 or more than 1 is clamped to 0 and 1
            target = torch.clamp(target, min=0.0, max=1.0)

            batched_tensor[order, :tensor_size_list[order] + pad_random, :] = target

            #           batched_tensor[order, :tensor_size_list[order], :] = target
            batched_ground_truth[order, :ground_truth_size_list[order]] = torch.tensor(ground_truth_list[order])

            # You do not need to know what loss mask is
            batched_loss_mask[order, :ground_truth_size_list[order]] = torch.ones(ground_truth_size_list[order])

            # logger.info('{}, {}, {}, {}'.format(batched_tensor.shape, batched_ground_truth.shape, batched_loss_mask.shape, ground_truth_size_list.shape))

        return [batched_tensor, batched_ground_truth, batched_loss_mask, ground_truth_size_list]

    def create_mel(self, wav_path):
        fs = 16000
        n_mels = 80
        frame_length_ms = 50
        frame_shift_ms = 25
        nsc = int(fs * frame_length_ms / 1000)
        nov = nsc - int(fs * frame_shift_ms / 1000)
        # nhop = int(fs * frame_shift_ms / 1000)
        eps = 1e-8
        db_ref = 160

        (rate, width, sig) = wavio.readwav(wav_path)
        y = sig.ravel()

        # logger.info('Shape of y: {}'.format(y.shape))

        # y, sr = librosa.core.load(wav_path, sr=fs)

        f, t, Zxx = sp.signal.stft(y, fs=fs, nperseg=nsc, noverlap=nov)

        # logger.info('After STFT')

        Sxx = np.abs(Zxx)

        # logger.info('Before librosa')

        mel_filters = librosa.filters.mel(sr=fs, n_fft=nsc, n_mels=n_mels)

        # logger.info('After librosa')

        mel_specgram = np.matmul(mel_filters, Sxx)

        # log10(0) is minus infinite, so replace mel_specgram values smaller than 'eps' as 'eps' (1e-8)
        log_mel_specgram = 20 * np.log10(np.maximum(mel_specgram, eps))

        # 20 * log10(eps) = 20 * -8 = -160
        # -160 is the smallest value
        # Add 160 and divide by 160 => Normalize value between 0 and 1
        norm_log_mel_specgram = (log_mel_specgram + db_ref) / db_ref

        # (F, T) -> (T, F)
        input_spectrogram = norm_log_mel_specgram.T
        # (T, F) -> (1, T, F)
        # Inserted the first axis to make stacking easier
        tensor_input = torch.tensor(input_spectrogram).view(1, input_spectrogram.shape[0], input_spectrogram.shape[1])
        return tensor_input


def train(net, optimizer, ctc_loss, input_tensor, ground_truth, loss_mask, target_lengths):
    # Shape of the input tensor (B, T, F)
    # B: Number of a batch (8, 16, or 64 ...)
    # T: Temporal length of an input
    # F: Number of frequency band, 80

    net.train()

    batch_size = input_tensor.shape[0]

    optimizer.zero_grad()

    # logger.info("Right before entering the model")

    pred_tensor = net(input_tensor)

    # Cast true sentence as Long data type, since CTC loss takes long tensor only
    # Shape (B, S)
    # S: Max length among true sentences
    truth = ground_truth
    # truth = truth.type(torch.cuda.LongTensor)
    truth = truth.type(torch.LongTensor)

    input_lengths = torch.full(size=(batch_size,), fill_value=pred_tensor.shape[0], dtype=torch.long)

    loss = ctc_loss(pred_tensor, truth, input_lengths, target_lengths)

    loss.backward()
    optimizer.step()

    # Return loss divided by true length because loss is sum of the character losses

    return pred_tensor, loss.item() / ground_truth.shape[1]


def evaluate(net, ctc_loss, input_tensor, ground_truth, loss_mask, target_lengths):
    # Shape of the input tensor (B, T, F)
    # B: Number of a batch (8, 16, or 64 ...)
    # T: Temporal length of an input
    # F: Number of frequency band, 80

    net.eval()

    batch_size = input_tensor.shape[0]

    pred_tensor = net(input_tensor)

    # Cast true sentence as Long data type, since CTC loss takes long tensor only
    # Shape (B, S)
    # S: Max length among true sentences
    truth = ground_truth
    # truth = truth.type(torch.cuda.LongTensor)
    truth = truth.type(torch.LongTensor)

    input_lengths = torch.full(size=(batch_size,), fill_value=pred_tensor.shape[0], dtype=torch.long)

    loss = ctc_loss(pred_tensor, truth, input_lengths, target_lengths)

    # Return loss divided by true length because loss is sum of the character losses

    return pred_tensor, loss.item() / ground_truth.shape[1]


def Decode_CTC_Prediction(prediction):
    CTC_pred = prediction.detach().cpu().numpy()
    result = list()
    last_elem = 0
    for i, elem in enumerate(CTC_pred):
        if elem != last_elem and elem != 0:
            result.append(elem)

        last_elem = elem

    result = np.asarray(result)

    return result


def main():

    global char2index
    global index2char
    global SOS_token
    global EOS_token
    global PAD_token

    parser = argparse.ArgumentParser(description='Speech hackathon Baseline')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden size of model (default: 256)')
    parser.add_argument('--layer_size', type=int, default=3, help='number of layers of model (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate in training (default: 0.2)')
    parser.add_argument('--bidirectional', action='store_true', help='use bidirectional RNN for encoder (default: False)')
    parser.add_argument('--use_attention', action='store_true', help='use attention between encoder-decoder (default: False)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training (default: 32)')
    parser.add_argument('--workers', type=int, default=4, help='number of workers in dataset loader (default: 4)')
    parser.add_argument('--max_epochs', type=int, default=10, help='number of max epochs in training (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-03, help='learning rate (default: 0.001)')
    parser.add_argument('--teacher_forcing', type=float, default=0.5, help='teacher forcing ratio in decoder (default: 0.5)')
    parser.add_argument('--max_len', type=int, default=80, help='maximum characters of sentence (default: 80)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--save_name', type=str, default='model', help='the name of model in nsml or local')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument("--pause", type=int, default=0)

    args = parser.parse_args()

    # No Problem in Label Handling

    char2index, index2char = label_loader.load_label('./hackathon.labels')
    SOS_token = char2index['<s>']  # '<sos>' or '<s>'
    EOS_token = char2index['</s>']  # '<eos>' or '</s>'
    PAD_token = char2index['_']  # '-' or '_'

    # My preprocessing

    unicode_jamo_list = list()

    # 초성
    for unicode in range(0x1100, 0x1113):
        unicode_jamo_list.append(chr(unicode))  # chr: Change hexadecimal to unicode
    # 중성
    for unicode in range(0x1161, 0x1176):
        unicode_jamo_list.append(chr(unicode))
    # 종성
    for unicode in range(0x11A8, 0x11C3):
        unicode_jamo_list.append(chr(unicode))
    for unicode in range(ord('A'), ord('Z') + 1):
        unicode_jamo_list.append(chr(unicode))
    for unicode in range(ord('a'), ord('z') + 1):
        unicode_jamo_list.append(chr(unicode))
    for unicode in range(ord('0'), ord('9') + 1):
        unicode_jamo_list.append(chr(unicode))

    unicode_jamo_list += [' ', '\\', '!', '~', '^', '<', '>', ',', '.', "'", '?', '？', '/', '%', '(', ')', ':', ';', '+',
                          '-', '<eos>']
    unicode_jamo_list.sort()
    # '_' symbol represents "blank" in CTC loss system, "blank" has to be the index 0
    unicode_jamo_list = ['_'] + unicode_jamo_list
    # Check the symbols
    logger.info(''.join(unicode_jamo_list))

    # End of my preprocessing

    tokenizer = Tokenizer(unicode_jamo_list)
    jamo_tokens = tokenizer.word2num(unicode_jamo_list)

    logger.info('Tokens: {}'.format(jamo_tokens))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    # N_FFT: defined in loader.py
    # feature_size = N_FFT / 2 + 1

    # <Model Define Block>

    # enc = EncoderRNN(feature_size, args.hidden_size,
    #                  input_dropout_p=args.dropout, dropout_p=args.dropout,
    #                  n_layers=args.layer_size, bidirectional=args.bidirectional, rnn_cell='gru', variable_lengths=False)
    #
    # dec = DecoderRNN(len(char2index), args.max_len, args.hidden_size * (2 if args.bidirectional else 1),
    #                  SOS_token, EOS_token,
    #                  n_layers=args.layer_size, rnn_cell='gru', bidirectional=args.bidirectional,
    #                  input_dropout_p=args.dropout, dropout_p=args.dropout, use_attention=args.use_attention)
    #
    # model = Seq2seq(enc, dec)

    net = Mel2SeqNet(80, 1024, 512, len(unicode_jamo_list), device)

    net_optimizer = optim.Adam(net.parameters(), lr=args.lr)

    ctc_loss = nn.CTCLoss().to(device)

    # <Model Define Block>

    # net.flatten_parameters()

    # for param in model.parameters():
    #     param.data.uniform_(-0.08, 0.08)

    # model = nn.DataParallel(model).to(device)

    # optimizer = optim.Adam(model.module.parameters(), lr=args.lr)
    # criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)

    # bind_model(model, optimizer)

    bind_model(net, net_optimizer)

    #

    if args.pause == 1:
        nsml.paused(scope=locals())

    if args.mode != "train":
        return

    # load data_list.csv    wav_046.wav   wav_046.label

    data_list = os.path.join(DATASET_PATH, 'train_data', 'data_list.csv')
    wav_paths = list()
    script_paths = list()
    korean_script_paths = list()

    with open(data_list, 'r') as f:
        for line in f:
            # line: "aaa.wav,aaa.label"

            wav_path, script_path = line.strip().split(',')
            korean_script_path = script_path.replace('.label', '.script')

            wav_paths.append(os.path.join(DATASET_PATH, 'train_data', wav_path))
            script_paths.append(os.path.join(DATASET_PATH, 'train_data', script_path))
            korean_script_paths.append(os.path.join(DATASET_PATH, 'train_data', korean_script_path))

    # Loading paths ended

    logger.info('Korean script path 0: {}'.format(korean_script_paths[0]))

    # Load Korean Scripts

    korean_script_list = list()
    jamo_script_list = list()

    jamo_regex = re.compile(u'[,_ ^.?!？~<>:;/%()+A-Za-z0-9\u1100-\u115e\u1161-\u11A7\u11a8-\u11ff]+')

    for file in korean_script_paths:
        with open(file, 'r') as f:
            line = f.read()
            line = line.strip()
            korean_script_list.append(line)
            jamo = jamotools.split_syllables(line, 'JAMO')
            jamo_filtered = ''.join(jamo_regex.findall(jamo))
            jamo_script_list.append(jamo_filtered)

    logger.info('Korean script 0: {}'.format(korean_script_list[0]))
    logger.info('Korean script 0 length: {}'.format(len(korean_script_list[0])))
    logger.info('Jamo script 0: {}'.format(jamo_script_list[0]))
    logger.info('Jamo script 0 length: {}'.format(len(jamo_script_list[0])))

    # ground_truth_list = list()
    # for i in range(len(jamo_script_list)):
    #     try:
    #         ground_truth_list.append(tokenizer.word2num(list(jamo_script_list[i]) + ['<eos>']))
    #     except:
    #         logger.info("Error found in string #{}: {}".format(i, jamo_script_list[i]))

    ground_truth_list = [(tokenizer.word2num(list(jamo_script_list[i]) + ['<eos>'])) for i in range(len(jamo_script_list))]

    # 90% of the data will be used as train
    split_index = int(0.9 * len(wav_paths))

    wav_path_list_train = wav_paths[:split_index]
    ground_truth_list_train = ground_truth_list[:split_index]

    wav_path_list_eval = wav_paths[split_index:]
    ground_truth_list_eval = ground_truth_list[split_index:]

    logger.info('Total:Train:Eval = {}:{}:{}'.format(len(wav_paths), len(wav_path_list_train), len(wav_path_list_eval)))

    batch_size = 16
    num_thread = 2

    preloader_eval = Threading_Batched_Preloader(wav_path_list_eval, ground_truth_list_eval, batch_size)
    preloader_train = Threading_Batched_Preloader(wav_path_list_train, ground_truth_list_train, batch_size)

    best_loss = 1e10
    begin_epoch = 0

    # load all target scripts for reducing disk i/o
    target_path = os.path.join(DATASET_PATH, 'train_label')
    load_targets(target_path)

    # train_batch_num, train_dataset_list, valid_dataset = split_dataset(args, wav_paths, script_paths, korean_script_paths, valid_ratio=0.05)

    logger.info('start')

    train_begin = time.time()

    for epoch in range(begin_epoch, args.max_epochs):

        logger.info((datetime.now().strftime('%m-%d %H:%M:%S')))

        preloader_eval.initialize_batch(num_thread)
        loss_list_eval = list()

        logger.info("Initialized Evaluation Preloader")

        count = 0
        while preloader_eval.end_flag == False:
            batch = preloader_eval.get_batch()
            if batch != None:
                tensor_input, ground_truth_, loss_mask, length_list = batch
                pred_tensor_, loss = evaluate(net, ctc_loss, tensor_input.to(device), ground_truth_.to(device),
                                              loss_mask.to(device), length_list.to(device))
                loss_list_eval.append(loss)

        eval_loss = np.mean(np.asarray(loss_list_eval))

        logger.info("Mean Evaluation Loss: {}".format(eval_loss))

        preloader_train.initialize_batch(num_thread)
        loss_list_train = list()

        logger.info("Initialized Training Preloader")

        count = 0
        while preloader_train.end_flag == False:
            batch = preloader_train.get_batch()
            # logger.info("Got Batch")
            if batch != None:
                # logger.info("Training Batch is not None")
                tensor_input, ground_truth, loss_mask, length_list = batch
                pred_tensor, loss = train(net, net_optimizer, ctc_loss, tensor_input.to(device),
                                          ground_truth.to(device), loss_mask.to(device), length_list.to(device))
                loss_list_train.append(loss)
                logger.info("Loss: {}".format(loss))
                count += 1
                logger.info("Train {}/{}".format(count, int(np.ceil(len(wav_path_list_train)/batch_size))))
                # logger.info("Training")
            else:
                logger.info("Training Batch is None")

        logger.info(loss_list_train)
        train_loss = np.mean(np.asarray(loss_list_train))

        logger.info("Mean Train Loss: {}".format(train_loss))

        # Start Training

        # train_queue = queue.Queue(args.workers * 2)
        #
        # train_loader = MultiLoader(train_dataset_list, train_queue, args.batch_size, args.workers)
        # train_loader.start()
        #
        # train_loss, train_cer = train(model, train_batch_num, train_queue, criterion, optimizer, device, train_begin, args.workers, 10, args.teacher_forcing)
        # logger.info('Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))
        #
        # train_loader.join()
        #
        # # Start Evaluation
        #
        # valid_queue = queue.Queue(args.workers * 2)
        # valid_loader = BaseDataLoader(valid_dataset, valid_queue, args.batch_size, 0)
        # valid_loader.start()
        #
        # eval_loss, eval_cer = evaluate(model, valid_loader, valid_queue, criterion, device)
        # logger.info('Epoch %d (Evaluate) Loss %0.4f CER %0.4f' % (epoch, eval_loss, eval_cer))
        #
        # valid_loader.join()

        # Evaluation Finished

        nsml.report(False,
            step=epoch, train_epoch__loss=train_loss, train_epoch__cer=0,
            eval__loss=eval_loss, eval__cer=0)

        # nsml.report(False,
        #     step=epoch, train_epoch__loss=train_loss, train_epoch__cer=train_cer,
        #     eval__loss=eval_loss, eval__cer=eval_cer)

        best_model = (eval_loss < best_loss)
        nsml.save(args.save_name)

        if best_model:
            nsml.save('best')
            best_loss = eval_loss

if __name__ == "__main__":
    main()
