import numpy as np
import matplotlib.pyplot as plt


def get_fimo_tsv(filepath):
    dataset = {}
    with open(filepath) as f:
        header = f.readline()
        print(header)
        while True:
            line = f.readline()
            if not line:
                break
            data = line.split()
            try:
                motif_alt_id = data[1]
                if motif_alt_id == 'Zfx':
                    motif_alt_id = 'ZFX'
                dataset[motif_alt_id] = dataset.get(motif_alt_id, 0) + 1
            except Exception as e:
                print(e)
                break
        f.close()
    return dataset


def sort_element_motifs(dataset):
    sorted_element_motifs = sorted(dataset.items(), key=lambda d: d[1], reverse=True)
    return sorted_element_motifs


def get_width_y(filename):
    print(filename)
    motifs = get_fimo_tsv(filename)
    # print(motifs)
    lines = 0
    for key in motifs:
        lines += motifs[key]
    # motif significant score
    motif_significant_score = {}
    for key in motifs:
        values = motifs[key]
        motif_significant_score[key] = -np.log(values / lines)
    sorted_motifs = sort_element_motifs(motif_significant_score)
    # print(sorted_motifs)
    width = [value[1] for value in sorted_motifs]
    y = [value[0] for value in sorted_motifs]
    width = width[-20:]
    y = y[-20:]
    print(width)
    print(y)
    return width, y


def figure10(y, width):
    rects = plt.barh(range(len(width)), y, align='center', color='#599AAD')
    plt.yticks(range(len(width)), width, font='Times New Roman', fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    for rect in rects:
        wd = rect.get_width()
        wdstr = ('%.3f' % (np.round(wd, 3)))
        wd = np.round(wd, 3)
        plt.xlim(0, 6)
        plt.text(wd, rect.get_y() + 0.5 / 2, wdstr, va='center', font='Times New Roman', fontsize=12)


if __name__ == '__main__':
    plt.figure(figsize=(20, 18))
    plt.rc('font', family='Times New Roman')
    plt.subplot(231)
    plt.text(-0.5, 21, "(A)", fontsize=20, fontweight='bold')
    plt.title('Predicted Silencers', fontsize=15)
    plt.xlabel('Motif Significant Score', font='Times New Roman', fontsize=15)
    y_predicted, width_predicted = get_width_y('data/fimo-predicted-silencers.tsv')
    figure10(y_predicted, width_predicted)

    plt.subplot(232)
    plt.text(-0.5, 21, "(B)", fontsize=20, fontweight='bold')
    plt.title('Silencers', fontsize=15)
    plt.xlabel('Motif Significant Score', font='Times New Roman', fontsize=15)
    y_real, width_real = get_width_y('data/fimo-silencers.tsv')
    figure10(y_real, width_real)

    plt.subplot(233)
    plt.text(-0.5, 21, "(C)", fontsize=20, fontweight='bold')
    plt.title('Exclusive Silencers compared to Annotation Models', fontsize=15)
    plt.xlabel('Motif Significant Score', font='Times New Roman', fontsize=15)
    y_exclusive, width_exclusive = get_width_y('data/fimo-exclusive.tsv')
    figure10(y_exclusive, width_exclusive)


    plt.subplot(234)
    plt.text(-0.5, 21, "(D)", fontsize=20, fontweight='bold')
    plt.title('ChromHMM', fontsize=15)
    plt.xlabel('Motif Significant Score', font='Times New Roman', fontsize=15)
    y_chromhmm, width_chromhmm = get_width_y('data/fimo-ChromHMM.tsv')
    figure10(y_chromhmm, width_chromhmm)

    plt.subplot(235)
    plt.text(-0.5, 21, "(E)", fontsize=20, fontweight='bold')
    plt.title('Segway', fontsize=15)
    plt.xlabel('Motif Significant Score', font='Times New Roman', fontsize=15)
    y_segway, width_segway = get_width_y('data/fimo-segway.tsv')
    figure10(y_segway, width_segway)

    plt.subplot(236)
    plt.text(-0.5, 21, "(F)", fontsize=20, fontweight='bold')
    plt.title("Libbrecht's Model", fontsize=15)
    plt.xlabel('Motif Significant Score', font='Times New Roman', fontsize=15)
    y_auto, width_auto = get_width_y('data/fimo-fullyAutomated.tsv')
    figure10(y_auto, width_auto)

    plt.savefig('Figure10.png', bbox_inches='tight', format='png', dpi=300)
    plt.show()
