import pybedtools


def get_bed(filename):
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
    print(len(bed))
    return bed


def get_strs(filename):
    strs = ''
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            fields = line.split()
            strs += f'{fields[0]}\t{fields[1]}\t{fields[2]}\n'
        f.close()
    return strs


def get_annotation_bed(filename):
    strs_silencers = get_strs(f'{filename}/silencer.txt')
    silencers_bed = pybedtools.BedTool(strs_silencers, from_string=True)

    strs_negative = get_strs(f'{filename}/negative.txt')
    negative_bed = pybedtools.BedTool(strs_negative, from_string=True)

    print(f'len(silencers_bed) = {len(silencers_bed)}, len(negative_bed) = {len(negative_bed)}\n'
          f'len(silencers_bed) + len(negative_bed) = {len(silencers_bed) + len(negative_bed)}')
    return silencers_bed, negative_bed


def get_overlapped_length(intersect):
    dict1 = {}
    for line in intersect:
        values = str(line).split()
        silencer = f'{values[0]}\t{values[1]}\t{values[2]}\n'
        start1 = int(values[1])
        end1 = int(values[2])
        start2 = int(values[4])
        end2 = int(values[5])
        length = min(end1, end2) - max(start1, start2)
        dict1[silencer] = dict1.get(silencer, 0) + length
    return dict1


def get_scores(real_silencers_bed, real_negative_bed, annotated_silencer_bed, annotated_negative_bed, filename):
    intersect1 = real_silencers_bed.intersect(annotated_silencer_bed, wa=True, wb=True)
    intersect2 = real_silencers_bed.intersect(annotated_negative_bed, wa=True, wb=True)
    dict1 = get_overlapped_length(intersect1)
    dict2 = get_overlapped_length(intersect2)

    intersect3 = real_negative_bed.intersect(annotated_silencer_bed, wa=True, wb=True)
    intersect4 = real_negative_bed.intersect(annotated_negative_bed, wa=True, wb=True)
    dict3 = get_overlapped_length(intersect3)
    dict4 = get_overlapped_length(intersect4)

    context = ''
    count = 0
    for line in real_silencers_bed:
        values = str(line).split()
        silencer = f'{values[0]}\t{values[1]}\t{values[2]}\n'
        pos_length = dict1.get(silencer, 0)
        neg_length = dict2.get(silencer, 0)
        if pos_length == 0 and neg_length == 0:
            count += 1
            context += f'{values[0]}-{values[1]}-{values[2]}\t1\t{0}\n'
            continue
        score = pos_length / (pos_length + neg_length)
        context += f'{values[0]}-{values[1]}-{values[2]}\t1\t{score}\n'

    count = 0
    for line in real_negative_bed:
        values = str(line).split()
        negative = f'{values[0]}\t{values[1]}\t{values[2]}\n'
        pos_length = dict3.get(negative, 0)
        neg_length = dict4.get(negative, 0)
        if pos_length == 0 and neg_length == 0:
            count += 1
            context += f'{values[0]}-{values[1]}-{values[2]}\t0\t0\n'
            continue
        score = pos_length / (pos_length + neg_length)
        context += f'{values[0]}-{values[1]}-{values[2]}\t0\t{score}\n'

    with open(filename, 'w') as f:
        f.write(context)
        f.close()


if __name__ == '__main__':
    real_silencers_filename = 'data/Figure7/silencers_in_REIN.txt'
    real_negative_filename = 'data/Figure7/negative_samples_in_REIN.txt'
    real_silencers = get_bed(real_silencers_filename)
    real_negative = get_bed(real_negative_filename)

    for anno in ['ChromHMM', 'Segway', 'FullyAutomated']:
        annotated_silencers, annotated_negative = get_annotation_bed(f'../../data/Annotations/{anno}/annotations-CREs')
        get_scores(real_silencers, real_negative,
                   annotated_silencers, annotated_negative,
                   f'data/Figure7/{anno}-score.txt')
