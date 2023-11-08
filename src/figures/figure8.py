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

with open('data/Figure8/ChromHMM.pkl', 'rb') as file:
    ans_chromhmm = pickle.load(file)
    print(ans_chromhmm)

with open('data/Figure8/segway.pkl', 'rb') as file:
    ans_segway2 = pickle.load(file)
    print(ans_segway2)

with open('data/Figure8/auto.pkl', 'rb') as file:
    ans_auto = pickle.load(file)
    print(ans_auto)


def get_color_mapping(txt):
    mapping = {}
    with open(txt) as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            anno = line.split()[0]
            mapping[anno] = index
        f.close()
    return mapping


fig = plt.figure(figsize=(15, 10))

rect1 = [0.10, 0.75, 0.8, 0.20]
rect2 = [0.10, 0.45, 0.8, 0.20]
rect3 = [0.10, 0.15, 0.8, 0.20]

ax1 = plt.axes(rect1)

plt.title('(A)', font='Times New Roman', fontsize=20, fontweight='bold', loc='left')
plt.xlim(0, 125)
plt.yticks([])
plt.xticks([])
plt.axis('off')
left = 0
i = 0
color_mapping = get_color_mapping('data/color_mapping/chromHMM.txt')
scale = 2
for item in ans_chromhmm:
    label, value = item
    color_idx = color_mapping[label]
    plt.barh(1, value * 100, left=left, label=label, color=colors[color_idx], edgecolor='grey', height=1)
    if i > 18:
        scale = 2.5
    if i > 20:
        scale = 2.8
    if i < 8:
        strs = ('%.1f' % (np.round(value * 100, 3))) + "%"
        plt.text(left + (value * 100) / 2 - 1.6, 0.85, strs, va='center', font='Times New Roman',
                 fontsize=14, rotation=60)
        bias = 0
        if label == 'EnhWF':
            bias = -0.2
        plt.text(left + (value * 100) / 2 - 1.8 + bias, 1.15, label, va='center', font='Times New Roman',
                 fontsize=14, rotation=60)
    else:
        x = [left + (value * 100) / 2, left + (value * 100) / 2 - (16 - i) * scale]
        y = [0.5, 0.3]
        plt.plot(x, y, linewidth=1, color=colors[color_idx])
        plt.plot(x, y, linewidth=1, color=colors[color_idx])

        strs = ('%.1f' % (np.round(value * 100, 3))) + "%"
        plt.text(x[1] - 0.5, y[1] - 0.1, strs, va='center', font='Times New Roman',
                 fontsize=14, rotation=90)
        plt.text(x[1] - 0.5, y[1] - 0.5, label, va='center', font='Times New Roman',
                 fontsize=14, rotation=90)

    left += value * 100
    i += 1
plt.subplots_adjust(bottom=0.4)  # 调整底部边距
# plt.legend(bbox_to_anchor=(0.5, -1.1), loc=8, ncol=8)
plt.tight_layout()

ax2 = plt.axes(rect2)

plt.title('(B)', font='Times New Roman', fontsize=20, fontweight='bold', loc='left')
plt.xlim(0, 125)
plt.yticks([])
plt.xticks([])
plt.axis('off')
left = 0
color_mapping = get_color_mapping('data/color_mapping/segway.txt')
i = 0
scale = 2
for item in ans_segway2:
    label, value = item
    color_idx = color_mapping[label]
    plt.barh(1, value * 100, left=left, label=label, color=colors[color_idx], edgecolor='grey', height=1)
    if i > 18:
        scale = 2.5
    if i > 20:
        scale = 3
    if i < 14:
        if i == 4:
            strs = ('%.1f' % (np.round(value * 100, 3))) + "%"
            plt.text(left + (value * 100) / 2 - 1.5, 0.85, strs, va='center', font='Times New Roman',
                     fontsize=14, rotation=60)
            plt.text(left + (value * 100) / 2 - 2.5, 1.15, label, va='center', font='Times New Roman',
                     fontsize=14, rotation=60)
        else:
            strs = ('%.1f' % (np.round(value * 100, 3))) + "%"
            plt.text(left + (value * 100) / 2 - 1.5, 0.85, strs, va='center', font='Times New Roman',
                     fontsize=14, rotation=60)
            plt.text(left + (value * 100) / 2 - 1.8, 1.15, label, va='center', font='Times New Roman',
                     fontsize=14, rotation=60)

    else:
        x = [left + (value * 100) / 2, left + (value * 100) / 2 - (19 - i) * scale]
        y = [0.5, 0.3]
        plt.plot(x, y, linewidth=1, color=colors[color_idx])
        plt.plot(x, y, linewidth=1, color=colors[color_idx])

        strs = ('%.1f' % (np.round(value * 100, 3))) + "%"
        plt.text(x[1] - 0.5, y[1] - 0.1, strs, va='center', font='Times New Roman',
                 fontsize=14, rotation=90)
        plt.text(x[1] - 0.5, y[1] - 0.5, label, va='center', font='Times New Roman',
                 fontsize=14, rotation=90)
    left += value * 100
    i += 1

plt.tight_layout()

ax3 = plt.axes(rect3)

plt.title("(C)", font='Times New Roman', fontsize=20, fontweight='bold', loc='left')
plt.xlim(0, 125)
plt.yticks([])
plt.xticks([])
plt.axis('off')
color_mapping = get_color_mapping('data/color_mapping/auto.txt')
left = 0
i = 0
scale = 5
for item in ans_auto:
    label, value = item
    color_idx = color_mapping[label]
    plt.barh(1, value * 100, left=left, label=label,
             color=colors[color_idx], edgecolor='grey', height=2
             )
    if i < 4:
        strs = ('%.1f' % (np.round(value * 100, 3))) + "%"
        plt.text(left + (value * 100) / 2 - 0.5, 0.85, strs, va='center', font='Times New Roman',
                 fontsize=14, rotation=60)
        plt.text(left + (value * 100) / 2 - 4.5, 1.15, label, va='center', font='Times New Roman',
                 fontsize=14, rotation=60)
    else:
        x = [left + (value * 100) / 2, left + (value * 100) / 2 - (6 - i) * scale]
        y = [0.0, -0.3]
        plt.plot(x, y, linewidth=1, color=colors[color_idx])
        plt.plot(x, y, linewidth=1, color=colors[color_idx])

        strs = ('%.1f' % (np.round(value * 100, 3))) + "%"
        plt.text(x[1] - 0.5, y[1] - 0.2, strs, va='center', font='Times New Roman',
                 fontsize=12, rotation=90)
        if label == 'ConstitutiveHet':
            plt.text(x[1] - 2.0, y[1] - 1.2, 'Constitutive-', va='center', font='Times New Roman',
                     fontsize=14, rotation=90)
            plt.text(x[1] + 0.0, y[1] - 1.2, 'Het', va='center', font='Times New Roman',
                     fontsize=14, rotation=90)
        elif label == 'LowConfidence':
            plt.text(x[1] - 2.0, y[1] - 1.2, 'Low-', va='center', font='Times New Roman',
                     fontsize=14, rotation=90)
            plt.text(x[1] + 0.0, y[1] - 1.2, 'Confidence', va='center', font='Times New Roman',
                     fontsize=14, rotation=90)
        else:
            plt.text(x[1] - 0.5, y[1] - 1.2, label, va='center', font='Times New Roman',
                     fontsize=14, rotation=90)
    left += value * 100
    i += 1
plt.barh(1, 0, left=left, label='',
         color=colors[i], edgecolor='grey',
         )
plt.tight_layout()

plt.savefig('Figure8.png', dpi=300)

plt.show()
