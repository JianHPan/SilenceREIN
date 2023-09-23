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
    motifs = get_fimo_tsv(filename)
    print(motifs)
    lines = 0
    for key in motifs:
        lines += motifs[key]
    # motif significant score
    motif_significant_score = {}
    for key in motifs:
        values = motifs[key]
        motif_significant_score[key] = -np.log(values / lines)
    sorted_motifs = sort_element_motifs(motif_significant_score)
    print(sorted_motifs)
    width = [value[1] for value in sorted_motifs]
    y = [value[0] for value in sorted_motifs]
    width = width[-20:]
    y = y[-20:]
    print(width)
    print(y)
    return width, y


def figure7(y, width):
    rects = plt.barh(range(len(width)), y, align='center', color='#599AAD')
    plt.yticks(range(len(width)), width, font='Times New Roman', fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    for rect in rects:
        wd = rect.get_width()
        wdstr = ('%.3f' % (np.round(wd, 3)))
        wd = np.round(wd, 3)
        plt.text(wd, rect.get_y() + 0.5 / 2, wdstr, va='center', font='Times New Roman', fontsize=12)


if __name__ == '__main__':
    plt.figure(figsize=(20, 10))
    plt.rc('font', family='Times New Roman')
    plt.subplot(121)
    plt.text(-0.5, 21, "(A)", fontsize=20, fontweight='bold')
    plt.title('Predicted Silencers', fontsize=15)
    plt.xlabel('Motif Significant Score', font='Times New Roman', fontsize=15)
    y_predicted, width_predicted = get_width_y('data/fimo-predicted-silencers.tsv')
    figure7(y_predicted, width_predicted)

    plt.subplot(122)
    plt.text(-0.5, 21, "(B)", fontsize=20, fontweight='bold')
    plt.title('Silencers', fontsize=15)
    plt.xlabel('Motif Significant Score', font='Times New Roman', fontsize=15)
    y_silencers, width_silencers = get_width_y('data/fimo-silencers.tsv')
    figure7(y_silencers, width_silencers)

    plt.savefig('Figure7.svg', bbox_inches='tight', format='svg', dpi=1200)
    plt.show()
