import csv
import json

lines = []
n_atoms = []
mol_index = []
atomic_number = {}
band_gaps = []


def main():
    global atomic_number
    pt_fp = "OMDB_data/periodic_table.csv"
    with open(pt_fp, 'r') as csv_pt:
        pt_reader = csv.DictReader(csv_pt, delimiter=',')
        for row in pt_reader:
            atomic_number[row['Symbol']] = int(row['AtomicNumber'])

    global band_gaps
    with open(r'OMDB_data/bandgaps.csv', 'r') as bgf:
        [band_gaps.append(line.strip()) for line in bgf.readlines()]

    fp = "OMDB_data/structures.xyz"
    global lines
    try:
        with open(fp, 'r') as f:
            lines = f.readlines()

    except:
        print(f'Error opening file: {fp}')
        exit(1)

    num_lines = len(lines)
    print(f'Number of lines = {num_lines}')
    global n_atoms
    global mol_index
    i = 0
    while i < len(lines):
        mol_index.append(i)
        try:
            j = int(lines[i].strip('\n'))
        except:
            print('Failed to parse number of atoms')
            exit()

        n_atoms.append(j)
        i += j + 2

    len_vectors = max(n_atoms) * 4
    with open("OMDB_data/omdb_vectors.csv", "w") as f:
        for idx in range(len(n_atoms)):
            v = get_vector(idx, len_vectors)
            s = json.dumps(v, separators=(' ', ':')).replace('[', '').replace(']', '').replace('\"', '')
            f.write(s + "\n")
            percent = '{:3.2f}'.format(idx / len(n_atoms) * 100)
            print('Progress:', percent + '%' + ' ' * (5 - len(percent)), end="\r")


def get_vector(idx, size):
    '''given the index for a molecule, generate the representation vector
	right-aligned zero-padded to the given size

	parameters
	----------
	idx  :  index of the molecule in the read file order

	size :  the length of the vector

	'''
    v = []
    for line in lines[mol_index[idx] + 2: mol_index[idx] + n_atoms[idx] + 2]:
        tok = line.split()
        symbol = tok[0]
        v.append(atomic_number[symbol])
        for t in tok[1:]:
            v.append(t)
    v.extend([0] * (size - len(v)))
    v.append(band_gaps[idx])
    return v


if __name__ == '__main__':
    main()
