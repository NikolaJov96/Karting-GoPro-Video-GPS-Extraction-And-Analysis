import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def main(out_dir, sector_files):
    """
    Generates a table comparing best sector times between two drivers
    """
    num_sectors = np.load(sector_files[0]).shape[0]
    num_drivers = len(sector_files)

    sectors_data = np.zeros((num_sectors, len(sector_files)))

    for driver_id, sector_file_path in enumerate(sector_files):
        file_sectors = np.load(sector_file_path)[:,-1].reshape(-1)
        assert file_sectors.shape[0] == num_sectors, 'Unequal number of sectors'
        sectors_data[:, driver_id] = file_sectors

    gaps = np.abs(sectors_data[:,0] - sectors_data[:,1])
    best_sectors = np.argmin(sectors_data, axis=1)
    total_laps = np.sum(sectors_data, axis=0)

    cell_text = [['Sector'] + [f'Driver {d + 1}' for d in range(num_drivers)] + ['Gap']]
    for i, sectors_data_row in enumerate(sectors_data):
        row = [str(i + 1)]
        row += ['{:.3f}'.format(s) for s in list(sectors_data_row)]
        row += ['{:.3f}'.format(gaps[i])]
        cell_text.append(row)
    total_row = ['Total'] + ['{:.3f}'.format(s) for s in total_laps]
    total_row += ['{:.3f}'.format(abs(total_laps[0] - total_laps[1]))]
    cell_text.append(total_row)

    colors = [['w' for _ in range(num_drivers + 2)] for _ in range(num_sectors + 2)]
    for sector_id in range(num_sectors):
        colors[sector_id + 1][best_sectors[sector_id] + 1] = '#00f000'

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.set_title('Best sector comparison')
    ax.axis('tight')
    ax.axis('off')
    ax.table(
        cellText=cell_text,
        cellColours=colors,
        loc='center'
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'sector_table.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str, help='Path to the output directory')
    parser.add_argument('first_sector_file', type=str, help='Path to the first sector time description file')
    parser.add_argument('second_sector_file', type=str,  help='Path to the second sector time description file')
    args = parser.parse_args()

    main(args.out_dir, [args.first_sector_file, args.second_sector_file])

