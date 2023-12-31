import os
import pickle

import pybedtools
import matplotlib.pyplot as plt
import numpy as np


def get_silencers(filename):
    strs = ''
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            fields = line.split()
            strs += f'{fields[0]}\t{fields[1]}\t{fields[2]}\n'
        f.close()
    bed = pybedtools.BedTool(strs, from_string=True)
    return bed


def get_annotation_chromHMM(filename):
    strs = ''
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            fields = line.split()
            strs += f'{fields[0]}\t{fields[1]}\t{fields[2]}\t{fields[3]}\n'
        f.close()
    return pybedtools.BedTool(strs, from_string=True)


def get_annotation_auto(filename):
    strs = ''
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            fields = line.split()
            cls = fields[3].split('_')[1]
            strs += f'{fields[0]}\t{fields[1]}\t{fields[2]}\t{cls}\n'
        f.close()
    return pybedtools.BedTool(strs, from_string=True)


def get_annotation_segway(filename):
    strs = ''
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            fields = line.split()
            cls = fields[3]
            strs += f'{fields[0]}\t{fields[1]}\t{fields[2]}\t{cls}\n'
        f.close()
    return pybedtools.BedTool(strs, from_string=True)


def get_annotation_with_mnemonics(filename, mapping_file):
    mapping = {}
    with open(mapping_file) as f:
        print(f'header: {f.readline()}')
        lines = f.readlines()
        for line in lines:
            code, cls = line.split()
            mapping[code] = cls
    strs = ''
    with open(filename) as f:
        if 'segway' in filename:
            print(f.readline())
        while True:
            line = f.readline()
            if not line:
                break
            fields = line.split()
            code = fields[3]
            cls = mapping[code]
            strs += f'{fields[0]}\t{fields[1]}\t{fields[2]}\t{cls}\n'
        f.close()
    return pybedtools.BedTool(strs, from_string=True)


# get_percent_fun_with_length
def get_percent_fun(silencers, annotation):
    print(f'len(silencers): {len(silencers)}')
    intersect1 = silencers.intersect(annotation, wa=True, wb=True)

    intersect = silencers.intersect(annotation, wo=True)

    print(f'-wa -wb : {len(intersect1)}, -wo : {len(intersect)}')
    lengths = {}

    for idx, line in enumerate(intersect):
        if idx == 0:
            print(line)
        values = str(line).split()
        cls = values[-2]
        length = int(values[-1])
        if length > 600:
            print('overlap > 600')
        lengths[cls] = lengths.get(cls, 0) + length

    all_lengths = {}
    counts = {}
    for line in annotation:
        fields = str(line).split()
        start = int(fields[1])
        end = int(fields[2])
        cls = fields[3]
        if end < start:
            print(f'{end} < {start}')
        all_lengths[cls] = all_lengths.get(cls, 0) + (end - start)
        counts[cls] = counts.get(cls, 0) + 1

    total_length = 0
    for key in lengths:
        total_length += lengths[key]
    lst = []
    for key in lengths:
        lst.append([key, lengths[key] / total_length])
    ans = sorted(lst, key=lambda x: x[1], reverse=True)
    print(ans)
    return ans


if __name__ == '__main__':
    silenceREIN_filename = '../../result/predict/silencers.txt'
    Silencers = get_silencers(silenceREIN_filename)

    # chrommhmm = get_annotation_chromHMM('../../data/Annotations/ChromHMM/ChromHMM-hg38.txt')
    # segway = get_annotation_segway('../../data/Annotations/Segway/segway_k562_hg38.bed')
    # auto = get_annotation_auto('../../data/Annotations/FullyAutomated/auto-hg38.bed')
    #
    # print('chromhmm-------------------------------------------')
    # ans_chromhmm = get_percent_fun(Silencers, chrommhmm)
    # print('segway-------------------------------------------')
    # ans_segway2 = get_percent_fun(Silencers, segway)
    # print('auto-------------------------------------------')
    # ans_auto = get_percent_fun(Silencers, auto)
    #
    # if not os.path.isdir('data/Figure8'):
    #     os.makedirs('data/Figure8')
    #
    # with open('data/Figure8/ChromHMM.pkl', 'wb') as file:
    #     pickle.dump(ans_chromhmm, file)
    #
    # with open('data/Figure8/segway.pkl', 'wb') as file:
    #     pickle.dump(ans_segway2, file)
    #
    # with open('data/Figure8/auto.pkl', 'wb') as file:
    #     pickle.dump(ans_auto, file)

    # annotation generated by pre-trained model
    flag = ''
    chrommhmm_with_mnemonics = get_annotation_with_mnemonics('../../data/Annotations/pre_trained/ChromHMM/K562_25_segments.bed',
                                                             f'../../data/Annotations/pre_trained/ChromHMM/mnemonics.txt')
    segway_with_mnemonics = get_annotation_with_mnemonics('../../data/Annotations/pre_trained/Segway/segway.bed',
                                                          f'../../data/Annotations/pre_trained/Segway/mnemonics.txt')
    print('chrommhmm_with_mnemonics-------------------------------------------')
    chrommhmm_with_mnemonics = get_percent_fun(Silencers, chrommhmm_with_mnemonics)
    print('segway_with_mnemonics------------------------------------------')
    segway_with_mnemonics = get_percent_fun(Silencers, segway_with_mnemonics)

    if not os.path.isdir('data/Figure9'):
        os.makedirs('data/Figure9')

    with open(f'data/Figure9/ChromHMM.pkl', 'wb') as file:
        pickle.dump(chrommhmm_with_mnemonics, file)

    with open(f'data/Figure9/segway.pkl', 'wb') as file:
        pickle.dump(segway_with_mnemonics, file)
