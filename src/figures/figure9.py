import matplotlib.pyplot as plt
import pickle
import numpy as np

colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
    '#b5cf6b', '#b3b3cc', '#f1b6da', '#cfcec4', '#dfc27d'
]


def get_color_mapping(txt):
    mapping = {}
    with open(txt) as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            anno = line.split()[0]
            mapping[anno] = index
        f.close()
    return mapping


colors_mapping = get_color_mapping('data/color_mapping/auto.txt')

with open(f'data/Figure9/ChromHMM.pkl', 'rb') as file:
    ans_chromhmm = pickle.load(file)
    print(ans_chromhmm)

with open(f'data/Figure9/segway.pkl', 'rb') as file:
    ans_segway = pickle.load(file)
    print(ans_segway)


fig = plt.figure(figsize=(15, 10))

rect1 = [0.10, 0.6, 0.9, 0.15]
rect2 = [0.10, 0.3, 0.9, 0.15]

ax1 = plt.axes(rect1)

plt.title('(A)', font='Times New Roman', fontsize=20, fontweight='bold', loc='left')
plt.xlim(0, 125)
plt.yticks([])
plt.xticks([])
plt.axis('off')
left = 0
i = 0

scale = 12
for item in ans_chromhmm:
    label, value = item
    color_idx = colors_mapping[label]
    plt.barh(1, value * 100, left=left, label=label, color=colors[color_idx], edgecolor='grey', height=1)
    if i < 3:
        strs = ('%.1f' % (np.round(value * 100, 3))) + "%"
        plt.text(left + (value * 100) / 2 - 0.2, 0.85, strs, va='center', font='Times New Roman',
                 fontsize=14, rotation=60)
        plt.text(left + (value * 100) / 2 - 3.5, 1.03, label, va='center', font='Times New Roman',
                 fontsize=14, rotation=60)
    else:
        x = [left + (value * 100) / 2, left + (value * 100) / 2 - (5.5 - i) * (scale + 5 - i)]
        y = [0.5, 0.3]
        plt.plot(x, y, linewidth=1, color=colors[color_idx])
        plt.plot(x, y, linewidth=1, color=colors[color_idx])
        strs = ('%.1f' % (np.round(value * 100, 3))) + "%"
        plt.text(x[1] - 0.5, y[1] - 0.1, strs, va='center', font='Times New Roman',
                 fontsize=12, rotation=0)
        plt.text(x[1] - 1.5, y[1] - 0.3, label, va='center', font='Times New Roman',
                 fontsize=14, rotation=0)

    left += value * 100
    i += 1
plt.subplots_adjust(bottom=0.4)
plt.tight_layout()


ax2 = plt.axes(rect2)

plt.title('(B)', font='Times New Roman', fontsize=20, fontweight='bold', loc='left')
plt.xlim(0, 125)
plt.yticks([])
plt.xticks([])
plt.axis('off')
left = 0
i = 0
scale = 10
for item in ans_segway:
    label, value = item
    color_idx = colors_mapping[label]
    plt.barh(1, value * 100, left=left, label=label, color=colors[color_idx], edgecolor='grey', height=1)
    if i < 3:
        strs = ('%.1f' % (np.round(value * 100, 3))) + "%"
        plt.text(left + (value * 100) / 2 - 0.2, 0.85, strs, va='center', font='Times New Roman',
                 fontsize=14, rotation=60)
        plt.text(left + (value * 100) / 2 - 3.5, 1.03, label, va='center', font='Times New Roman',
                 fontsize=14, rotation=60)
    else:
        x = [left + (value * 100) / 2, left + (value * 100) / 2 - (4.5 - i) * (scale + 5 - i)]
        y = [0.5, 0.3]
        plt.plot(x, y, linewidth=1, color=colors[color_idx])
        plt.plot(x, y, linewidth=1, color=colors[color_idx])
        strs = ('%.1f' % (np.round(value * 100, 3))) + "%"
        plt.text(x[1] - 0.5, y[1] - 0.1, strs, va='center', font='Times New Roman',
                 fontsize=12, rotation=0)
        bias = 0
        if 'Low' in label:
            bias = -2.0
        plt.text(x[1] - 1.5 + bias, y[1] - 0.3, label, va='center', font='Times New Roman',
                 fontsize=14, rotation=0)
    left += value * 100
    i += 1

plt.tight_layout()

plt.savefig(f'Figure9.png', dpi=300)

plt.show()
