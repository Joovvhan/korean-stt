#-*- coding: utf-8 -*-

import wavio
import queue
import threading
import random
import time
import torch
import logging
import torch.nn as nn

import copy
import scipy as sp
import numpy as np
import librosa
import sys
import os

import re
import jamotools

import Levenshtein as Lev

import psutil

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)

SAMPLE_RATE = 16000
frame_length_ms = 50
frame_shift_ms = 25
nsc = int(SAMPLE_RATE * frame_length_ms / 1000)
nov = nsc - int(SAMPLE_RATE * frame_shift_ms / 1000)
N_FFT = nsc
eps = 1e-8
db_ref = 160

target_dict = dict()

num_mels = 80

MEL_FILTERS = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=nsc, n_mels=num_mels)



# Baseline Function
def load_targets(path):
    with open(path, 'r') as f:
        for no, line in enumerate(f):
            key, target = line.strip().split(',')  # wav_001,192 755 662 192 678 476 662 408 690 2 125 610 662 220 640 125 662 179 192 661 123 662
            target_dict[key] = target


# Baseline Function
def get_script(filepath, bos_id, eos_id):
    key = filepath.split('/')[-1].split('.')[0]
    script = target_dict[key]
    tokens = script.split(' ')
    result = list()
    result.append(bos_id)
    for i in range(len(tokens)):
        if len(tokens[i]) > 0:
            result.append(int(tokens[i]))
    result.append(eos_id)
    return result


# Baseline Function
def get_spectrogram_feature(filepath):
    (rate, width, sig) = wavio.readwav(filepath)
    sig = sig.ravel()

    stft = torch.stft(torch.FloatTensor(sig),
                        N_FFT,
                        hop_length=int(0.01*SAMPLE_RATE),
                        win_length=int(0.030*SAMPLE_RATE),
                        window=torch.hamming_window(int(0.030*SAMPLE_RATE)),
                        center=False,
                        normalized=False,
                        onesided=True)

    stft = (stft[:, :, 0].pow(2) + stft[:, :, 1].pow(2)).pow(0.5)
    amag = stft.numpy()
    feat = torch.FloatTensor(amag)

    feat = torch.FloatTensor(feat).transpose(0, 1)

    return feat


# Baseline Function
def load_label(label_path):
    char2index = dict() # [ch] = id
    index2char = dict() # [id] = ch
    with open(label_path, 'r') as f:
        for no, line in enumerate(f):
            if line[0] == '#':
                continue
            index, char, freq = line.strip().split('\t')
            char = char.strip()
            if len(char) == 0:
                char = ' '
            char2index[char] = int(index)
            index2char[int(index)] = char

    return char2index, index2char


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
    def __init__(self, wav_path_list, ground_truth_list, korean_script_list, batch_size, num_mels):
        super(Threading_Batched_Preloader).__init__()
        self.wav_path_list = wav_path_list
        self.total_num_input = len(wav_path_list)
        self.tensor_input_list = [None] * self.total_num_input
        self.ground_truth_list = ground_truth_list
        self.korean_script_list = korean_script_list
        self.sentence_length_list = np.asarray(list(map(len, ground_truth_list)))
        self.shuffle_step = 12
        self.loading_sequence = None
        self.end_flag = False
        self.batch_size = batch_size
        self.qsize = 8
        self.queue = queue.Queue(self.qsize)
        self.thread_flags = list()
        self.num_mels = num_mels

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
        logger.info(thread_size)
        load_idxs_list = list()
        for i in range(thread_num):
            start_idx = i * thread_size
            end_idx = (i + 1) * thread_size

            if end_idx > loading_sequence_len:
                end_idx = loading_sequence_len

            load_idxs_list.append(loading_sequence[start_idx:end_idx])

        self.end_flag = False

        self.queue = queue.Queue(self.qsize)
        self.thread_flags = [False] * thread_num

        self.thread_list = [
            Batching_Thread(self.wav_path_list, self.ground_truth_list, self.korean_script_list, load_idxs_list[i], self.queue, self.batch_size,
                            self.thread_flags, self.num_mels, i) for i in range(thread_num)]

        for thread in self.thread_list:
            thread.start()
        logger.info('batch initialized')
        return

    def check_thread_flags(self):
        for flag in self.thread_flags:
            if flag is False:
                return False

        logger.info("All Threads Finished")

        if self.queue.empty:
            self.end_flag = True
            logger.info("Queue is Empty")
            for thread in self.thread_list:
                thread.join()
            return True

        return False

    def get_batch(self):
        # logger.info("Called get_batch ")
        # logger.info(self.queue)
        while not (self.check_thread_flags()):
            #logger.info("wErwer")
            # logger.info(psutil.virtual_memory())
            # logger.info("Q Size Before: {}".format(self.queue.qsize()))
            batch = self.queue.get()
            # logger.info("Q Size After: {}".format(self.queue.qsize()))
            # logger.info(psutil.virtual_memory())
            if (batch != None):
                #logger.info("Batch is Ready")
                batched_tensor = batch[0]
                batched_ground_truth = batch[1]
                batched_loss_mask = batch[2]
                ground_truth_size_list = batch[3]
                korean_script_list = batch[4]

                return batched_tensor, batched_ground_truth, batched_loss_mask, ground_truth_size_list, korean_script_list

            else:
                logger.info("Loader Queue is empty")
                #time.sleep(1)

        logger.info('get_batch finish')
        return None


class Batching_Thread(threading.Thread):

    def __init__(self, wav_path_list, ground_truth_list, korean_script_list, load_idxs_list, queue, batch_size, thread_flags, num_mels, id):

        threading.Thread.__init__(self)
        self.wav_path_list = wav_path_list
        self.ground_truth_list = ground_truth_list
        self.korean_script_list = korean_script_list
        self.load_idxs_list = load_idxs_list
        self.list_len = len(load_idxs_list)
        self.cur_idx = 0
        self.id = id
        self.queue = queue
        self.batch_size = batch_size
        self.thread_flags = thread_flags
        self.num_mels = num_mels

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
        korean_script_list = list()

        count = 0
        max_seq_len = 0
        max_sen_len = 0

        for i in range(self.batch_size):

            # If there is no more file, break and set end_flag true
            if self.cur_idx >= self.list_len:
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

            korean_script_list.append(self.korean_script_list[self.load_idxs_list[self.cur_idx]])

            self.cur_idx += 1
            count += 1

        batched_tensor = torch.zeros(count, max_seq_len + 5, self.num_mels)
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

        return [batched_tensor, batched_ground_truth, batched_loss_mask, ground_truth_size_list, korean_script_list]

    def create_mel(self, wav_path):
        fs = 16000
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

        # Cut-off paddings
        coef = np.sum(Sxx, 0)
        Sxx = Sxx[:, find_starting_point(coef):find_ending_point(coef)]

        # logger.info('Before librosa')

        # logger.info('After librosa')

        mel_specgram = np.matmul(MEL_FILTERS, Sxx)

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
        tensor_input = torch.tensor(input_spectrogram).unsqueeze(0)
        return tensor_input


def train(net, optimizer, ctc_loss, input_tensor, ground_truth, target_lengths, device):
    # Shape of the input tensor (B, T, F)
    # B: Number of a batch (8, 16, or 64 ...)
    # T: Temporal length of an input
    # F: Number of frequency band, 80

    batch_size = input_tensor.shape[0]

    optimizer.zero_grad()

    # logger.info("Right before entering the model")

    pred_tensor = net(input_tensor)

    # Cast true sentence as Long data type, since CTC loss takes long tensor only
    # Shape (B, S)
    # S: Max length among true sentences
    truth = ground_truth
    # truth = truth.type(torch.cuda.LongTensor)
    truth = truth.type(torch.LongTensor).to(device)

    input_lengths = torch.full(size=(batch_size,), fill_value=pred_tensor.shape[0], dtype=torch.long)

    loss = ctc_loss(pred_tensor, truth, input_lengths, target_lengths)

    loss.backward()
    optimizer.step()

    # Return loss divided by true length because loss is sum of the character losses

    return pred_tensor, loss.item() / ground_truth.shape[1]


def evaluate(net, ctc_loss, input_tensor, ground_truth, target_lengths, device):
    # Shape of the input tensor (B, T, F)
    # B: Number of a batch (8, 16, or 64 ...)
    # T: Temporal length of an input
    # F: Number of frequency band, 80

    batch_size = input_tensor.shape[0]

    pred_tensor = net(input_tensor)

    # Cast true sentence as Long data type, since CTC loss takes long tensor only
    # Shape (B, S)
    # S: Max length among true sentences
    truth = ground_truth
    # truth = truth.type(torch.cuda.LongTensor)
    truth = truth.type(torch.LongTensor).to(device)

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


def My_Unicode_Jamo():
    # 초성

    unicode_jamo_list = list()
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
                          '-', '<s>', '</s>']
    unicode_jamo_list.sort()
    # '_' symbol represents "blank" in CTC loss system, "blank" has to be the index 0
    unicode_jamo_list = ['_'] + unicode_jamo_list

    return unicode_jamo_list


def get_paths(DATASET_PATH):
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

    return wav_paths, script_paths, korean_script_paths


def get_korean_and_jamo_list(korean_script_paths):

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

    return korean_script_list, jamo_script_list


def Decode_Prediction(pred_tensor, tokenizer):
    decoded_list = list()
    for i in range(pred_tensor.shape[1]):
        _, CTC_index = pred_tensor[:, i, :].max(-1)
        index = Decode_CTC_Prediction(CTC_index)
        jamos = tokenizer.num2word(index)
        sentence = jamotools.join_jamos(''.join(jamos))

        not_com_jamo = re.compile(u'[^\u3130-\u3190]')
        filtered_sentence = ''.join(not_com_jamo.findall(sentence))
        filtered_sentence = filtered_sentence.replace('<s>', '')
        filtered_sentence = filtered_sentence.replace('</s>', '')

        decoded_list.append(filtered_sentence)
    return decoded_list


def lev_num_to_lev_string(lev_num_list, index2char):
    lev_str_list = list()
    for num_list in lev_num_list:

        temp = list()
        for num in num_list:
            temp.append(index2char[num])

        lev_str_list.append(''.join(temp))

    return lev_str_list


def char_distance(ref, hyp):
    ref = ref.replace(' ', '')
    hyp = hyp.replace(' ', '')

    dist = Lev.distance(hyp, ref)
    length = len(ref.replace(' ', ''))

    return dist, length


def char_distance_list(ref_list, hyp_list):
    sum_dist = 0
    sum_length = 0

    for ref, hyp in zip(ref_list, hyp_list):
        dist, length = char_distance(ref, hyp)
        sum_dist += dist
        sum_length += length

    return sum_dist, sum_length


def find_starting_point(coef, thres=0.1, margin=10):
    starting_point = 0
    for i in range(len(coef) - 1):
        if (coef[i] <= thres and coef[i + 1] > thres):
            starting_point = i
            break

    starting_point = starting_point - margin

    if starting_point < 0:
        starting_point = 0

    return starting_point


def find_ending_point(coef, thres=0.1, margin=10):

    ending_point = len(coef) - 1

    for i in range(len(coef) - 1, 0, -1):
        if (coef[i] <= thres and coef[i - 1] > thres):
            ending_point = i
            break

    ending_point = ending_point + margin

    if ending_point > len(coef):
        ending_point = len(coef)

    return ending_point
