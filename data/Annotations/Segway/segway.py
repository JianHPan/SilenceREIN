import os


def write(dict_, keys, filename):
    count_ = 0
    print(keys)
    with open(filename, 'w') as fw:
        for k in keys:
            count_ += len(dict_[k])
            for l1 in dict_[k]:
                fw.write(l1)
        fw.close()
    print(count_)


if __name__ == '__main__':
    dataset = {}
    cls = set()
    with open('segway_k562_hg38.bed') as f:
        while True:
            line = f.readline()
            if not line:
                break
            fields = line.split()
            key = fields[3]
            cls.add(key)
            dataset.setdefault(key, []).append(f'{fields[0]}\t{fields[1]}\t{fields[2]}\n')
        f.close()

    silencer_count = 0
    negative_count = 0

    silencer_keys = ['Repr2', 'Repr4', 'Repr3', 'Repr1']
    negative_keys = []
    for i in dataset.keys():
        if i in silencer_keys:
            continue
        negative_keys.append(i)

    if not os.path.isdir('annotations-CREs'):
        os.makedirs('annotations-CREs')
    write(dataset, silencer_keys, 'annotations-CREs/silencer.txt')
    write(dataset, negative_keys, 'annotations-CREs/negative.txt')

