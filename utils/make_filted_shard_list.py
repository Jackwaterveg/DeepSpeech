#!/usr/bin/env python3

# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import io
import logging
import os
import tarfile
import time
import multiprocessing
import random
import torch
import torchaudio
import torchaudio.backend.sox_io_backend as sox

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


def write_tar_file(data_list,
                   no_segments,
                   tar_file,
                   resample=16000,
                   index=0,
                   total=1):
    logging.info('Processing {} {}/{}'.format(tar_file, index, total))
    read_time = 0.0
    save_time = 0.0
    write_time = 0.0
    with tarfile.open(tar_file, "w") as tar:
        prev_wav = None
        for item in data_list:
            if no_segments:
                key, txt, wav = item
            else:
                key, txt, wav, start, end = item

            suffix = wav.split('.')[-1]
            assert suffix in AUDIO_FORMAT_SETS
            if no_segments:
                ts = time.time()
                with open(wav, 'rb') as fin:
                    data = fin.read()
                read_time += (time.time() - ts)
            else:
                if wav != prev_wav:
                    ts = time.time()
                    waveforms, sample_rate = sox.load(wav, normalize=False)
                    read_time += (time.time() - ts)
                    prev_wav = wav
                start = int(start * sample_rate)
                end = int(end * sample_rate)
                audio = waveforms[:1, start:end]

                # resample
                if sample_rate != resample:
                    audio = torchaudio.transforms.Resample(
                        sample_rate, resample)(audio)

                ts = time.time()
                f = io.BytesIO()
                sox.save(f, audio, resample, format="wav", bits_per_sample=16)
                # Save to wav for segments file
                suffix = "wav"
                f.seek(0)
                data = f.read()
                save_time += (time.time() - ts)

            assert isinstance(txt, str)
            ts = time.time()
            txt_file = key + '.txt'
            txt = txt.encode('utf8')
            txt_data = io.BytesIO(txt)
            txt_info = tarfile.TarInfo(txt_file)
            txt_info.size = len(txt)
            tar.addfile(txt_info, txt_data)

            wav_file = key + '.' + suffix
            wav_data = io.BytesIO(data)
            wav_info = tarfile.TarInfo(wav_file)
            wav_info.size = len(data)
            tar.addfile(wav_info, wav_data)
            write_time += (time.time() - ts)
        logging.info('read {} save {} write {}'.format(read_time, save_time,
                                                       write_time))

def data_filter(data,
           max_length=1200,
           min_length=10,
           token_max_length=100,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, wav, label, sample_rate}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    #for sample in data:
    key, txt, wav, start, end = data
    num_frames = (end -start) * 100
    
    # sample['wav'] is torch.Tensor, we have 100 frames every second
    if num_frames < min_length:
        return False
    if num_frames > max_length:
        return False
    if len(txt) < token_min_length:
        return False
    if len(txt) > token_max_length:
        return False
    if num_frames != 0:
        if len(txt) / num_frames < min_output_input_ratio:
            return False
        if len(txt) / num_frames > max_output_input_ratio:
            return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_node',
                        type=int,
                        default=1,
                        help='number of node')
    parser.add_argument('--num_gpus_per_node',
                        type=int,
                        default=8,
                        help='number of gpus per node')
    parser.add_argument('--num_utts_per_shard',
                        type=int,
                        default=1000,
                        help='num utts per shard')
    parser.add_argument('--do_filter',
                        action='store_true',
                        default=False,
                        help='whether to filter the data according to the wav length and the text length')
    parser.add_argument('--num_threads',
                        type=int,
                        default=1,
                        help='num threads for make shards')
    parser.add_argument('--prefix',
                        default='shards',
                        help='prefix of shards tar file')
    parser.add_argument('--segments', default=None, help='segments file')
    parser.add_argument('--resample',
                        type=int,
                        default=16000,
                        help='segments file')
    parser.add_argument('wav_file', help='wav file')
    parser.add_argument('text_file', help='text file')
    parser.add_argument('shards_dir', help='output shards dir')
    parser.add_argument('shards_list', help='output shards list file')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    torch.set_num_threads(1)
    wav_table = {}
    with open(args.wav_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            wav_table[arr[0]] = arr[1]

    no_segments = True
    segments_table = {}
    if args.segments is not None:
        no_segments = False
        with open(args.segments, 'r', encoding='utf8') as fin:
            for line in fin:
                arr = line.strip().split()
                assert len(arr) == 4
                segments_table[arr[0]] = (arr[1], float(arr[2]), float(arr[3]))

    data = []
    with open(args.text_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split(maxsplit=1)
            key = arr[0]
            txt = arr[1] if len(arr) > 1 else ''
            if no_segments:
                assert key in wav_table
                wav = wav_table[key]
                data.append((key, txt, wav))
            else:
                wav_key, start, end = segments_table[key]
                wav = wav_table[wav_key]
                data_piece = (key, txt, wav, start, end)
                if args.do_filter:
                    data_piece = data_filter(data_piece)
                    data.append(data_piece)
                else:
                    continue
    
    num = args.num_utts_per_shard
    # 多少个 tar 包
    tar_num = int(len(data) / num)
    # 每个卡多少个 tar 包
    num_cu = args.num_node * args.num_gpus_per_node
    each_cu_tar_tum = int(tar_num / num_cu)
    need_num = each_cu_tar_tum * num_cu * num
    remove_num = len(data) - need_num
    logging.info(f"remove {remove_num} data piece")
    remove_stride = int(len(data) / remove_num)
    logging.info(f"remove stride {remove_stride}")
    remove_index = range(len(data))[0::remove_stride]
    
    remove_index = remove_index[0: remove_num]
    assert(len(remove_index) == remove_num)
   # remove_index = [random.randint(0,len(len(data))) for i in range(remove_num)]
    remove_index = sorted(remove_index)
    for i in reversed(remove_index):
        del data[i]
    logging.info(f"number of save data {len(data)}")
    #data = data[:need_num]

    chunks = [data[i:i + num] for i in range(0, len(data), num)]


    os.makedirs(args.shards_dir, exist_ok=True)

    # Using thread pool to speedup
    pool = multiprocessing.Pool(processes=args.num_threads)
    shards_list = []
    tasks_list = []
    num_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        tar_file = os.path.join(args.shards_dir,
                                '{}_{:09d}.tar'.format(args.prefix, i))
        shards_list.append(tar_file)
        pool.apply_async(
            write_tar_file,
            (chunk, no_segments, tar_file, args.resample, i, num_chunks))

    pool.close()
    pool.join()

    with open(args.shards_list, 'w', encoding='utf8') as fout:
        for name in shards_list:
            fout.write(name + '\n')
