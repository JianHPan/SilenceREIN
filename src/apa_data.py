import pybedtools


def get_anchors(filenames):
    anchors = set()
    for filename in filenames:
        with open(filename) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if line[0] == '#':
                    # print(line)
                    continue
                values = line.split()
                chrom1, start1, end1 = values[0], values[1], values[2]
                chrom2, start2, end2 = values[3], values[4], values[5]
                anchors.add(f'{chrom1}-{start1}-{end1}')
                anchors.add(f'{chrom2}-{start2}-{end2}')
    return anchors


def get_promoter(filename, left, right):
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
    bed = bed.slop(genome='hg38', l=left, r=right)
    print(f'{filename}: '
          f' number of normalized coordinates is {len(bed)}')
    return bed


def get_silencer(filename):
    coordinates = set()
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            values = line.split()
            coordinates.add(f'{values[0]}\t{values[1]}\t{values[2]}\n')
        f.close()
    print(f"number of elements in '{filename}': {len(coordinates)}")
    strs = ''
    for coordinate in coordinates:
        strs += coordinate
    bed = pybedtools.BedTool(strs, from_string=True)
    bed = bed.sort()
    print(f'{filename}: '
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
        filename = CREs2Filename[CRE_cls]
        if CRE_cls == 'silencer':
            bed2 = get_silencer(filename)
        elif CRE_cls == 'promoter':
            bed2 = get_promoter(filename, left=300, right=299)
        else:
            exit(0)

        bed1_and_bed2 = bed2.intersect(bed1, wa=True, wb=True, f=0.5)
        tmp = {}
        t = set()
        for idx, line in enumerate(bed1_and_bed2):
            values = str(line).split()
            tmp.setdefault(
                f'{values[3]}-{values[4]}-{values[5]}', []).append(
                f'{values[0]}-{values[1]}-{values[2]}'
            )
            t.add(f'{values[0]}-{values[1]}-{values[2]}')
        dataset[CRE_cls] = tmp
        print(f'number of {CRE_cls}: {len(t)}')
    return dataset


def get_peaks_file(loop_filenames, relationship, flag):
    peaks = 'chr1\tx1\tx2\tchr2\ty1\ty2\tcolor\tcomment\n'
    contexts = []
    for filename in loop_filenames:
        with open(filename) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if line[0] == '#':
                    # print(line)
                    continue
                values = line.split()
                chrom1, start1, end1 = values[0], values[1], values[2]
                chrom2, start2, end2 = values[3], values[4], values[5]
                anchor1 = f'{chrom1}-{start1}-{end1}'
                anchor2 = f'{chrom2}-{start2}-{end2}'
                # s-p link
                silencers = relationship['silencer'].get(anchor1, [])
                promoters = relationship['promoter'].get(anchor2, [])
                if len(silencers) > 0 and len(promoters) > 0:
                    for silencer in silencers:
                        chrom_silencer, start_silencer, end_silencer = silencer.split('-')
                        for promoter in promoters:
                            chrom_promoter, start_promoter, end_promoter = promoter.split('-')
                            context = f'{chrom_silencer}\t{start_silencer}\t{end_silencer}\t' \
                                      f'{chrom_promoter}\t{start_promoter}\t{end_promoter}\t' \
                                      f'0,255,0\tnon\n'
                            if context not in contexts:
                                contexts.append(context)
                # p-s link
                promoters = relationship['promoter'].get(anchor1, [])
                silencers = relationship['silencer'].get(anchor2, [])
                if len(promoters) > 0 and len(silencers) > 0:
                    for promoter in promoters:
                        chrom_promoter, start_promoter, end_promoter = promoter.split('-')
                        for silencer in silencers:
                            chrom_silencer, start_silencer, end_silencer = silencer.split('-')
                            context = f'{chrom_promoter}\t{start_promoter}\t{end_promoter}\t' \
                                      f'{chrom_silencer}\t{start_silencer}\t{end_silencer}\t' \
                                      f'0,255,0\tnon\n'
                            if context not in contexts:
                                contexts.append(context)
    for context in contexts:
        peaks += context
    with open(f'../APA/Peaks/Peaks-{flag}.bedpe', 'w') as f:
        f.write(peaks)
        f.close()

#

if __name__ == '__main__':
    file1 = f'../data/ChIA-PET/K562/ENCFF030PMM.css.t1.sig.bedpe'
    file2 = f'../data/ChIA-PET/K562/ENCFF118PBQ.css.t1.sig.bedpe'
    file3 = f'../data/ChIA-PET/K562/ENCFF511QFN.css.t1.sig.bedpe'
    file4 = f'../data/ChIA-PET/K562/ENCFF607PZX.css.t1.sig.bedpe'
    file5 = f'../data/ChIA-PET/K562/ENCFF759YBZ.css.t1.sig.bedpe'

    ctcf_loops = [file2, file4, ]

    antibody = 'CTCF'
    print(f'{antibody}------------------------------------------------------------------------')
    CREs2Filename = {
        'silencer': '../data/APA/CRE/NB/silencer-hg38.bed',
        'promoter': '../data/CREs/K562/promoters.bed'
    }
    ctcf_anchors = get_anchors(ctcf_loops)
    ctcf_relation = get_relationship(ctcf_anchors)
    get_peaks_file(ctcf_loops, ctcf_relation, f'NB-{antibody}')

    # ----------------------------------------
    CREs2Filename['silencer'] = '../data/APA/CRE/correlation_SVM/silencer-correlation-hg38.bed'
    ctcf_anchors = get_anchors(ctcf_loops)
    ctcf_relation = get_relationship(ctcf_anchors)
    get_peaks_file(ctcf_loops, ctcf_relation, f'correlation-{antibody}')

    # ----------------------------------------
    CREs2Filename['silencer'] = '../data/APA/CRE/correlation_SVM/silencer-SVM-hg38.bed'
    ctcf_anchors = get_anchors(ctcf_loops)
    ctcf_relation = get_relationship(ctcf_anchors)
    get_peaks_file(ctcf_loops, ctcf_relation, f'SVM-{antibody}')

    # ----------------------------------------
    CREs2Filename['silencer'] = '../data/APA/CRE/SilenceREIN/predict-silencers.txt'
    ctcf_anchors = get_anchors(ctcf_loops)
    ctcf_relation = get_relationship(ctcf_anchors)
    get_peaks_file(ctcf_loops, ctcf_relation, f'SilenceREIN-{antibody}')

    # ----------------------------------------
    CREs2Filename['silencer'] = '../data/APA/CRE/gkmSVM/silencers-hg38.txt'
    ctcf_anchors = get_anchors(ctcf_loops)
    ctcf_relation = get_relationship(ctcf_anchors)
    get_peaks_file(ctcf_loops, ctcf_relation, f'gkmSVM-{antibody}')

    CREs2Filename['silencer'] = '../data/APA/CRE/DeepSilencer/silencers-hg38.txt'
    ctcf_anchors = get_anchors(ctcf_loops)
    ctcf_relation = get_relationship(ctcf_anchors)
    get_peaks_file(ctcf_loops, ctcf_relation, f'DeepSilencer-{antibody}')


    CREs2Filename['silencer'] = '../data/APA/CRE/CNN/silencers-hg38.txt'
    ctcf_anchors = get_anchors(ctcf_loops)
    ctcf_relation = get_relationship(ctcf_anchors)
    get_peaks_file(ctcf_loops, ctcf_relation, f'CNN-{antibody}')
    # ---------------------------------------------------------------------

    polr2a_loops = [file1, file3, file5]
    antibody = 'POLR2A'
    print(f'{antibody}------------------------------------------------------------------------')

    CREs2Filename = {
        'silencer': '../data/APA/CRE/NB/silencer-hg38.bed',
        'promoter': '../data/CREs/K562/promoters.bed'
    }
    polr2a_anchors = get_anchors(polr2a_loops)
    polr2a_relation = get_relationship(polr2a_anchors)
    get_peaks_file(polr2a_loops, polr2a_relation, f'NB-{antibody}')

    # ----------------------------------------
    CREs2Filename['silencer'] = '../data/APA/CRE/correlation_SVM/silencer-correlation-hg38.bed'
    polr2a_anchors = get_anchors(polr2a_loops)
    polr2a_relation = get_relationship(polr2a_anchors)
    get_peaks_file(polr2a_loops, polr2a_relation, f'correlation-{antibody}')

    # ----------------------------------------
    CREs2Filename['silencer'] = '../data/APA/CRE/correlation_SVM/silencer-SVM-hg38.bed'
    polr2a_anchors = get_anchors(polr2a_loops)
    polr2a_relation = get_relationship(polr2a_anchors)
    get_peaks_file(polr2a_loops, polr2a_relation, f'SVM-{antibody}')

    # ----------------------------------------
    CREs2Filename['silencer'] = '../data/APA/CRE/SilenceREIN/predict-silencers.txt'
    polr2a_anchors = get_anchors(polr2a_loops)
    polr2a_relation = get_relationship(polr2a_anchors)
    get_peaks_file(polr2a_loops, polr2a_relation, f'SilenceREIN-{antibody}')

    # ----------------------------------------
    CREs2Filename['silencer'] = '../data/APA/CRE/gkmSVM/silencers-hg38.txt'
    polr2a_anchors = get_anchors(polr2a_loops)
    polr2a_relation = get_relationship(polr2a_anchors)
    get_peaks_file(polr2a_loops, polr2a_relation, f'gkmSVM-{antibody}')

    # ----------------------------------------
    CREs2Filename['silencer'] = '../data/APA/CRE/DeepSilencer/silencers-hg38.txt'
    polr2a_anchors = get_anchors(polr2a_loops)
    polr2a_relation = get_relationship(polr2a_anchors)
    get_peaks_file(polr2a_loops, polr2a_relation, f'DeepSilencer-{antibody}')

    # ----------------------------------------
    CREs2Filename['silencer'] = '../data/APA/CRE/CNN/silencers-hg38.txt'
    polr2a_anchors = get_anchors(polr2a_loops)
    polr2a_relation = get_relationship(polr2a_anchors)
    get_peaks_file(polr2a_loops, polr2a_relation, f'CNN-{antibody}')


    # ---------------------------------------------------------------------
    h3k27ac_loops = ['../data/HiChIP/K562/SRR5831491.interactions.all.mango-meet',
                     '../data/HiChIP/K562/SRR5831492.interactions.all.mango-meet',
                     '../data/HiChIP/K562/SRR5831493.interactions.all.mango-meet']
    antibody = 'H3K27ac'
    print(f'{antibody}------------------------------------------------------------------------')

    CREs2Filename = {
        'silencer': '../data/APA/CRE/NB/silencer-hg38.bed',
        'promoter': '../data/CREs/K562/promoters.bed'
    }
    h3k27ac_anchors = get_anchors(h3k27ac_loops)
    h3k27ac_relation = get_relationship(h3k27ac_anchors)
    get_peaks_file(h3k27ac_loops, h3k27ac_relation, f'NB-{antibody}')

    # ----------------------------------------
    CREs2Filename['silencer'] = '../data/APA/CRE/correlation_SVM/silencer-correlation-hg38.bed'
    h3k27ac_anchors = get_anchors(h3k27ac_loops)
    h3k27ac_relation = get_relationship(h3k27ac_anchors)
    get_peaks_file(h3k27ac_loops, h3k27ac_relation, f'correlation-{antibody}')

    # ----------------------------------------
    CREs2Filename['silencer'] = '../data/APA/CRE/correlation_SVM/silencer-SVM-hg38.bed'
    h3k27ac_anchors = get_anchors(h3k27ac_loops)
    h3k27ac_relation = get_relationship(h3k27ac_anchors)
    get_peaks_file(h3k27ac_loops, h3k27ac_relation, f'SVM-{antibody}')

    # ----------------------------------------
    CREs2Filename['silencer'] = '../data/APA/CRE/SilenceREIN/predict-silencers.txt'
    h3k27ac_anchors = get_anchors(h3k27ac_loops)
    h3k27ac_relation = get_relationship(h3k27ac_anchors)
    get_peaks_file(h3k27ac_loops, h3k27ac_relation, f'SilenceREIN-{antibody}')

    # ----------------------------------------
    CREs2Filename['silencer'] = '../data/APA/CRE/gkmSVM/silencers-hg38.txt'
    h3k27ac_anchors = get_anchors(h3k27ac_loops)
    h3k27ac_relation = get_relationship(h3k27ac_anchors)
    get_peaks_file(h3k27ac_loops, h3k27ac_relation, f'gkmSVM-{antibody}')

    # ----------------------------------------
    CREs2Filename['silencer'] = '../data/APA/CRE/DeepSilencer/silencers-hg38.txt'
    h3k27ac_anchors = get_anchors(h3k27ac_loops)
    h3k27ac_relation = get_relationship(h3k27ac_anchors)
    get_peaks_file(h3k27ac_loops, h3k27ac_relation, f'DeepSilencer-{antibody}')

    # ----------------------------------------
    CREs2Filename['silencer'] = '../data/APA/CRE/CNN/silencers-hg38.txt'
    h3k27ac_anchors = get_anchors(h3k27ac_loops)
    h3k27ac_relation = get_relationship(h3k27ac_anchors)
    get_peaks_file(h3k27ac_loops, h3k27ac_relation, f'CNN-{antibody}')

