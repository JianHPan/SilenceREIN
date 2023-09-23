import argparse
import copy
import pybedtools
import json
import numpy as np
from tqdm import tqdm
import re
from utils import save_as_pickle
from myclass import REIN, Node
import os

nan = float('nan')
work_dir = os.getcwd()
hg38 = f'{work_dir}/data/hg38/hg38.fa'
threshold_length = 1000


# Expand silencer, promoter, enhancer, non-silencer all to 600bp
# Store all one-hot codes, histone ChIP-seq and TF ChIP-seq data (p-value), for 600bp DNA sequences
# CRE overlap by more than 50% of themselves to be considered as an element


def save_as_json(dataset, filepath):
    context = json.dumps(dataset, sort_keys=False, indent=4, separators=(',', ': '))
    with open(filepath, 'w') as save_f:
        save_f.write(context)
        save_f.close()


def get_loops(filenames):
    """
    Read chromatin loops from a 'bedpe' format files.txt and store them in dictionary format.
    Only chromatin loops with both anchor points less than 1000bp in length are retained
    :param filenames: a list of filenames of 'bedpe' format files.txt
    :return: chromatin loops in dictionary format
    """
    dataset = {}
    anchors = set()
    anchors_length = []
    count = 0
    for filename in filenames:
        with open(filename) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if line[0] == '#':
                    print(line)
                    continue
                values = line.split()
                chrom1, start1, end1 = values[0], values[1], values[2]
                chrom2, start2, end2 = values[3], values[4], values[5]
                start1 = int(start1)
                end1 = int(end1)
                start2 = int(start2)
                end2 = int(end2)
                # Only chromatin loops with both anchor points less than 1000bp in length are retained
                if (end1 - start1 > threshold_length) or (end2 - start2) > threshold_length:
                    continue
                count += 1

                # print(f'anchor1 length: {end1 - start1}, anchor2 length: {end2 - start2}')
                anchor1 = f'{chrom1}-{start1}-{end1}'
                anchor2 = f'{chrom2}-{start2}-{end2}'
                dataset.setdefault(anchor1, []).append(anchor2)
                dataset.setdefault(anchor2, []).append(anchor1)

                if anchor1 not in anchors:
                    anchors.add(anchor1)
                    anchors_length.append(end1 - start1)
                if anchor2 not in anchors:
                    anchors.add(anchor2)
                    anchors_length.append(end2 - start2)
            f.close()
    print(f'number of loops with high-resolution is: {count}')
    print(f'number of anchors with high-resolution is: {len(dataset.values())}')
    print(f'median length of anchors (less than {threshold_length}bp) is: {np.median(anchors_length)}')
    print(f'average length of anchors (less than {threshold_length}bp) is: {np.mean(anchors_length)}')
    print(f'longest length of anchors (less than {threshold_length}bp) is: {np.max(anchors_length)}')
    print(f'shortest length of anchors (less than {threshold_length}bp) is: {np.min(anchors_length)}')
    return dataset


def get_anchor2index(filenames):
    """
    The "merge" command in pybedtools was used to merge overlapping anchors.
    The "cluster" command in pybedtools was then used to obtain the cluster ids of the clustered anchors.
    These cluster ids correspond one-to-one with the merged anchors,
    allowing us to establish the correspondence between the coordinates of the original anchors and the indices of the merged anchors
    :param filenames: a list of filenames of 'bedpe' format files.txt
    :return: (1) a list of original anchors
             (2) a list of anchors (normalised anchors) all 600bp in length. A length of 600 bp was obtained by extending from the midpoint of the merged anchors towards both ends.
             (3) the mapping relationship between the coordinates of the original anchors and the ordinal numbers of the normalized anchors.
                 for example, {'chr1-0-600': 1,
                               'chr1-1000-1600': 1,
                               'chr2-0-600': 2,
                               ...}
    """
    anchors = set()
    # (1) Read the anchor from the .bedpe file and convert it to string format
    for filename in filenames:
        with open(filename) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if line[0] == '#':
                    print(line)
                    continue
                values = line.split()
                chrom1, start1, end1 = values[0], values[1], values[2]
                chrom2, start2, end2 = values[3], values[4], values[5]
                start1 = int(start1)
                end1 = int(end1)
                start2 = int(start2)
                end2 = int(end2)
                # Only chromatin loops with both anchor points less than 1000bp in length are retained
                if (end1 - start1 > threshold_length) or (end2 - start2) > threshold_length:
                    continue
                anchor1 = f'{chrom1}\t{start1}\t{end1}\n'
                anchor2 = f'{chrom2}\t{start2}\t{end2}\n'
                anchors.add(anchor1)
                anchors.add(anchor2)
            f.close()
    strs = ''
    for anchor in anchors:
        strs += anchor
    bed = pybedtools.BedTool(strs, from_string=True)
    # (2) sort
    anchors_sorted = bed.sort()
    # (3) cluster
    anchors = []
    anchor2index = {}
    anchors_cluster = anchors_sorted.cluster()
    for line in anchors_cluster:
        values = str(line).split()
        anchor = f'{values[0]}-{values[1]}-{values[2]}'
        cluster = int(values[3])
        anchor2index[anchor] = cluster
    # (4) merge. Also obtain the midpoints of the merged anchors.
    anchor_merged = anchors_sorted.merge()
    strs2 = ''
    lengths = []
    for line in anchor_merged:
        values = str(line).split()
        anchor = f'{values[0]}-{values[1]}-{values[2]}'
        lengths.append(int(values[2]) - int(values[1]))
        anchors.append(anchor)
        mid = int((int(values[1]) + int(values[2])) / 2)
        strs2 += f'{values[0]}\t{mid}\t{mid + 1}\n'
    print(f'number of node: {len(anchor2index)}, {len(anchors)}')
    # (5) Convert the length of the anchor point to 600
    bed2 = pybedtools.BedTool(strs2, from_string=True)
    anchor_slop = bed2.slop(genome='hg38', l=300, r=299)
    anchors_normalized = []
    for line in anchor_slop:
        values = str(line).split()
        anchor = f'{values[0]}-{values[1]}-{values[2]}'
        anchors_normalized.append(anchor)
    print(f'before and after normalized: {len(anchors)}, {len(anchors_normalized)}')

    print(f'number of merged-anchors is: {len(lengths)}')
    print(f'median length of merged-anchors (less than {threshold_length}bp) is: {np.median(lengths)}')
    print(f'average length of merged-anchors (less than {threshold_length}bp) is: {np.mean(lengths)}')
    print(f'longest length of merged-anchors (less than {threshold_length}bp) is: {np.max(lengths)}')
    print(f'shortest length of merged-anchors (less than {threshold_length}bp) is: {np.min(lengths)}')

    return anchors, anchors_normalized, anchor2index


def get_cis_regulatory_elements(filename, left, right):
    """
    Read the file in bed format via pybedtools and obtain CRE of 600bp in length
    :param filename: the filename of .bed file
    :param left: the length of extension to the left
    :param right: the length of extension to the right
    :return:  CRE of 600bp in length
    """
    coordinates = set()
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            values = line.split()
            mid = int((int(values[1]) + int(values[2])) / 2)
            coordinates.add(f'{values[0]}\t{mid}\t{mid + 1}\n')
        f.close()
    print(f"number of elements in '{filename}': {len(coordinates)}")
    strs = ''
    for coordinate in coordinates:
        strs += coordinate
    bed = pybedtools.BedTool(strs, from_string=True)
    bed = bed.sort()
    raw = copy.copy(bed)
    # bed = bed.slop(g='../data/fa/hg38.chrom.sizes', l=left, r=right)
    # https://daler.github.io/pybedtools/autodocs/pybedtools.bedtool.BedTool.slop.html
    bed = bed.slop(genome='hg38', l=left, r=right)
    print(f'{filename}: '
          f'number of raw coordinates is {len(raw)},'
          f' number of normalized coordinates is {len(bed)}')
    return bed


def get_relationship(anchors):
    strs1 = ''
    for anchor in anchors:
        chrom, start, end = anchor.split('-')
        strs1 += f'{chrom}\t{start}\t{end}\n'
    bed1 = pybedtools.BedTool(strs1, from_string=True)
    dataset = {}
    for CRE_cls in CREs2Filename:
        # Obtain coordinates of regulatory elements and assemble them into strings
        filename = CREs2Filename[CRE_cls]
        bed2 = get_cis_regulatory_elements(filename, left=300, right=299)
        # DNA sequences of CRE from the hg38 genome were obtained and preserved
        fn = pybedtools.BedTool(hg38)
        bed2.sequence(fi=fn, fo=f'{work_dir}/data/tmp/{CRE_cls}.fa')
        # intersect: get the relationship between anchors and CRE
        bed1_and_bed2 = bed2.intersect(bed1, wa=True, wb=True, f=0.5)
        # bed1_and_bed2 = bed1.intersect(bed2, wa=True, wb=True)
        # Recording transcriptional regulatory elements with overlapping anchors
        tmp = {}
        t = set()
        for idx, line in enumerate(bed1_and_bed2):
            values = str(line).split()
            # tmp.setdefault(f'{values[0]}-{values[1]}-{values[2]}', []).append(
            #     f'{values[3]}-{values[4]}-{values[5]}'
            # )
            # t.add(f'{values[3]}-{values[4]}-{values[5]}')
            tmp.setdefault(
                f'{values[3]}-{values[4]}-{values[5]}', []).append(
                f'{values[0]}-{values[1]}-{values[2]}'
            )
            t.add(f'{values[0]}-{values[1]}-{values[2]}')
        dataset[CRE_cls] = tmp
        # save_as_json(tmp, f'../data/tmp/anchors-and-{CRE_cls}.json')
        print(f'number of {CRE_cls}: {len(t)}')
    return dataset


def get_coordinate2seq(anchors, anchors_normalized):
    # (1) Slicing DNA sequences from the hg38 genome
    fn = pybedtools.BedTool(hg38)
    strs = ''
    for anchor in anchors:
        chrom, start, end = anchor.split('-')
        strs += f'{chrom}\t{start}\t{end}\n'
    bed = pybedtools.BedTool(strs, from_string=True)
    bed.sequence(fi=fn, fo=f'{work_dir}/data/tmp/tmp.fa')
    # (2) converting to string
    strs2 = ''
    for anchor in anchors_normalized:
        chrom, start, end = anchor.split('-')
        strs2 += f'{chrom}\t{start}\t{end}\n'
    bed = pybedtools.BedTool(strs2, from_string=True)
    bed.sequence(fi=fn, fo=f'{work_dir}/data/tmp/tmp_normalized.fa')
    # (3) Converting to dictionary format while reading fasta files from CRE
    coordinate2seq = {}
    for filename in ['silencer', 'enhancer', 'promoter', 'non-silencer', 'tmp', 'tmp_normalized']:
        with open(f'{work_dir}/data/tmp/{filename}.fa') as f:
            while True:
                coordinate = f.readline()
                if not coordinate:
                    break
                chromosome = re.search(r">(.*):", coordinate).group(1)
                start = re.search(r":(.*)-", coordinate).group(1)
                end = re.search(r"-(.*)", coordinate).group(1)
                seq = f.readline().split()[0]
                coordinate2seq[f'{chromosome}-{start}-{end}'] = seq
            f.close()
    return coordinate2seq


def fun(anchors, anchors_normalized, relationship):
    # Obtaining DNA sequences
    coordinate2seq = get_coordinate2seq(anchors, anchors_normalized)

    nodes = []
    count = len(anchors)
    print(f'count : {count}')
    for idx in tqdm(range(count), desc=f"build nodes ...", leave=False):
        anchor = anchors[idx]
        anchor_normalized = anchors_normalized[idx]
        node = Node()
        node.node_index = idx
        node.coordinate = anchor_normalized
        seq = coordinate2seq[node.coordinate]
        node.seq = seq

        # Access to all types of transcriptional regulatory elements that overlap with this node
        cis_regulatory_elements = {}
        for CRE_cls in relationship:
            cis_regulatory_elements[CRE_cls] = []
            for CRE_coordinate in relationship[CRE_cls].get(anchor, []):
                # Normalized coordinates
                cre = Node()
                cre.coordinate = CRE_coordinate
                seq = coordinate2seq[CRE_coordinate]
                cre.seq = seq

                cis_regulatory_elements[CRE_cls].append(cre)
        node.cis_regulatory_elements = cis_regulatory_elements
        nodes.append(node)
    return nodes


def build_nodes(anchors, anchors_normalized):
    print('prepare nodes')
    # Overlap with known transcriptional regulatory elements
    relationship = get_relationship(anchors)
    nodes = fun(anchors, anchors_normalized, relationship)
    return nodes


def build(filenames):
    # Obtaining high-resolution chromatin loops
    loops = get_loops(filenames)
    # Get mapping of original anchors, normalized anchors, anchors to cluster ids
    anchors, anchors_normalized, anchor2index = get_anchor2index(filenames)

    print('build nodes')
    nodes = build_nodes(anchors, anchors_normalized)

    print('build adjacency_list')
    edges = {}
    for node in anchor2index:
        # Get node ID
        index = anchor2index[node]
        neighbors = loops[node]
        for neighbor in neighbors:
            neighbor_index = anchor2index[neighbor]
            edges.setdefault(index, []).append(neighbor_index)
    # Convert dictionary format to list format and remove duplicates
    # save_as_json(edges, 'edges.json')
    node_num = len(edges)
    adj = []
    for idx in range(1, node_num + 1):
        neighbors = sorted(list(set(edges[idx])))
        neighbors = list(map(lambda x: x - 1, neighbors))
        adj.append(neighbors)
    graph = REIN(adj, nodes)
    return graph


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='input your info')
    parse.add_argument('--cell', type=str, default='K562')
    parse.add_argument('--HiC', type=str, default='ChIA-PET')
    args = parse.parse_args()

    print(work_dir)

    cell = args.cell
    HiC = args.HiC

    CREs2Filename = {
        'silencer': f'{work_dir}/data/CREs/{cell}/silencers.bed',
        'enhancer': f'{work_dir}/data/CREs/{cell}/enhancers.bed',
        'promoter': f'{work_dir}/data/CREs/{cell}/promoters.bed',
        'non-silencer': f'{work_dir}/data/CREs/{cell}/non-silencers.bed'
    }

    files = []
    if HiC == 'ChIA-PET':
        if cell == 'K562':
            file1 = f'{work_dir}/data/ChIA-PET/K562/ENCFF030PMM.css.t1.sig.bedpe'
            file2 = f'{work_dir}/data/ChIA-PET/K562/ENCFF118PBQ.css.t1.sig.bedpe'
            file3 = f'{work_dir}/data/ChIA-PET/K562/ENCFF511QFN.css.t1.sig.bedpe'
            file4 = f'{work_dir}/data/ChIA-PET/K562/ENCFF607PZX.css.t1.sig.bedpe'
            file5 = f'{work_dir}/data/ChIA-PET/K562/ENCFF759YBZ.css.t1.sig.bedpe'
            files = [file1, file2, file3, file4, file5]
        elif cell == 'HepG2':
            file1 = f'{work_dir}/data/ChIA-PET/HepG2/ENCFF115TNW.css.t1.sig.bedpe'
            file2 = f'{work_dir}/data/ChIA-PET/HepG2/ENCFF299NHM.css.t1.sig.bedpe'
            file3 = f'{work_dir}/data/ChIA-PET/HepG2/ENCFF360QPK.css.t1.sig.bedpe'
            file4 = f'{work_dir}/data/ChIA-PET/HepG2/ENCFF364UNM.css.t1.sig.bedpe'
            file5 = f'{work_dir}/data/ChIA-PET/HepG2/ENCFF743ZWY.css.t1.sig.bedpe'
            files = [file1, file2, file3, file4, file5]
        else:
            exit(0)
    elif HiC == 'HiChIP':
        if cell == 'K562':
            threshold_length = float('inf')
            file1 = f'{work_dir}/data/HiChIP/K562/SRR5831491.interactions.all.mango-meet'
            file2 = f'{work_dir}/data/HiChIP/K562/SRR5831492.interactions.all.mango-meet'
            file3 = f'{work_dir}/data/HiChIP/K562/SRR5831493.interactions.all.mango-meet'
            files = [file1, file2, file3]
            print('for HiChIP, we save all interactions')
        else:
            exit(0)

    if len(files) == 0:
        exit(0)
    Regulatory_Element_Interaction_Network = build(files)
    save_as_pickle(f'{work_dir}/data/REIN/{cell}/graph-{HiC}-check.pkl', Regulatory_Element_Interaction_Network, False)

