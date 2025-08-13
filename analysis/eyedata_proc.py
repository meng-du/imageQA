import os
import pandas as pd
import numpy as np
import pickle as pkl
from PIL import Image, ImageDraw
import seaborn as sns
from tqdm import tqdm


SCREEN_SIZE = (1920, 1080)


def process_eyetribe_data(df: pd.DataFrame, data: dict):
    img_starts = df.index[df['state'].str.contains('Showing image')]
    img_ends = df.index[df['state'].str.contains('End of image display')]
    assert len(img_starts) == len(img_ends)

    for i in range(len(img_starts)):
        start, end = img_starts[i], img_ends[i]
        img_name = df.iloc[start]['state'].split()[-1]
        block_name = df.iloc[start]['state'].split('_')[0] if img_name.startswith('im') else 'ex'
        img_df = df.iloc[start+1:end]
        img_df = img_df[img_df['state'] == '7']

        if img_df.empty:
            print(f'No data for {img_name} in {block_name}')
            continue

        img_data = {
            'fix': np.array(img_df['fix'].tolist()),
            'rawx': np.array(img_df['rawx'].tolist()),
            'rawy': np.array(img_df['rawy'].tolist()),
            'avgx': np.array(img_df['avgx'].tolist()),
            'avgy': np.array(img_df['avgy'].tolist())
        }
        assert len(img_data['fix']) == len(img_data['rawx']) == len(img_data['rawy']) == len(img_data['avgx']) == len(img_data['avgy'])

        if img_name in data:
            data[img_name][block_name] = img_data
        else:
            data[img_name] = {block_name: img_data}

    return data


def process_all_eyetribe_data():
    data = {}
    for sid in range(2, 7):
        data[sid] = {}
        for pt in range(1, 5):
            print(f'subject {sid} - part {pt}')
            try:
                df = pd.read_csv(f'/Users/me/repos/bio_ann/imageQA/log/{sid}pt{pt}_eyetribe.tsv', sep='\t')
            except Exception as e:
                print(f'{sid}pt{pt}_eyetribe.tsv: {e}')
                continue
            process_eyetribe_data(df, data[sid])
    with open(f'/Users/me/repos/bio_ann/imageQA/analysis/eyedata.pkl', 'wb') as f:
        pkl.dump(data, f)


def draw_on_image(eye_data, img_path, coord_type='raw', fix=False):
    with open(eye_data, 'rb') as f:
        eye_data = pkl.load(f)
    color_palettes = ['light:#006d2c', 'light:#08519c', 'light:#a63603', 'light:#980043', 'light:#54278f']

    image_names = set()
    for sid in eye_data:
        image_names.update(eye_data[sid].keys())
    for imgname in image_names:
        img = Image.open(os.path.join(img_path, imgname))
        img = img.resize((SCREEN_SIZE[0], img.size[1] * SCREEN_SIZE[0] // img.size[0]))
        new_img = Image.new('RGB', SCREEN_SIZE, color='black')
        new_img.paste(img, (0, (SCREEN_SIZE[1] - img.size[1]) // 2))
        for block in eye_data[3][imgname].keys():
            for s, sid in enumerate(eye_data):
                try:
                    data = eye_data[sid][imgname][block]
                except KeyError:
                    print(f'No data for subject {sid}, {imgname}, block {block}, {coord_type}')
                    continue
                if fix:
                    indices = np.where(data['fix'] == 'True')[0]
                else:
                    indices = np.arange(len(data[coord_type + 'x']))
                colors = sns.color_palette(color_palettes[s], len(indices)+5).as_hex()
                colors = colors[5:] # remove white ones
                draw = ImageDraw.Draw(new_img)
                for i, idx in enumerate(indices):
                    x = data[coord_type + 'x'][idx]
                    y = data[coord_type + 'y'][idx]
                    draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=colors[i], width=3)
            # save
            im = imgname.split('.')[0].split('/')[-1]
            b = block.split('_')[0]
            new_img.save(os.path.join(img_path, 'analysis', 'images', f'{im}_{b}_{coord_type}.jpg'))


def reformat4vipllava(coord_type='avg', fix=False):
    X_START = (SCREEN_SIZE[0] - SCREEN_SIZE[1]) // 2
    X_END = X_START + SCREEN_SIZE[1]
    UNIT_LENGTH = SCREEN_SIZE[1] / 24  # 24x24 Vipllava attention maps

    eye_data = pkl.load(open('eyedata.pkl', 'rb'))
    attn = {}
    for sid in tqdm(eye_data):
        for imgname in eye_data[sid]:
            im = imgname.split('/')[-1].split('.')[0]
            attn[im] = {}
            for block in eye_data[sid][imgname]:
                attn[im][block] = np.zeros((24, 24))
                data = eye_data[sid][imgname][block]
                coords = np.array([*zip(data[coord_type + 'x'], data[coord_type + 'y'])])
                for i, coord in enumerate(coords):
                    if fix and data['fix'][i] != 'True':
                        continue
                    x, y = coord
                    if X_START <= x <= X_END and 0 <= y <= SCREEN_SIZE[1]:
                        xi = int((x - X_START) // UNIT_LENGTH)
                        yi = int(y // UNIT_LENGTH)
                        attn[im][block][yi, xi] += 1
    fixname = 'fix' if fix else 'all'
    with open(f'eyedata_reformatted4vipllava_{coord_type}_{fixname}.pkl', 'wb') as f:
        pkl.dump(attn, f)


def aggregate(coord_type='avg', fix=False):
    UNIT_LENGTH = 24

    eye_data = pkl.load(open('/Users/me/repos/bio_ann/imageQA/analysis/eyedata.pkl', 'rb'))
    attn = {}
    for sid in tqdm(eye_data):
        for imgname in eye_data[sid]:
            im = imgname.split('/')[-1].split('.')[0]
            attn[im] = {}
            for block in eye_data[sid][imgname]:
                attn[im][block] = np.zeros((SCREEN_SIZE[1] // UNIT_LENGTH, SCREEN_SIZE[0] // UNIT_LENGTH))
                data = eye_data[sid][imgname][block]
                coords = np.array([*zip(data[coord_type + 'x'], data[coord_type + 'y'])])
                for i, coord in enumerate(coords):
                    if fix and data['fix'][i] != 'True':
                        continue
                    x, y = coord
                    if 0 <= x <= SCREEN_SIZE[0] and 0 <= y <= SCREEN_SIZE[1]:
                        xi = int(x // UNIT_LENGTH)
                        yi = int(y // UNIT_LENGTH)
                        attn[im][block][yi, xi] += 1
    fixname = 'fix' if fix else 'all'
    with open(f'eyedata_aggregated_{coord_type}_{fixname}.pkl', 'wb') as f:
        pkl.dump(attn, f)



if __name__ == '__main__':
    # process_all_eyetribe_data()
    # draw_on_image('/Users/me/repos/bio_ann/imageQA/analysis/eyedata.pkl', '/Users/me/repos/bio_ann/imageQA/', 'raw', fix=False)
    # draw_on_image('/Users/me/repos/bio_ann/imageQA/analysis/eyedata.pkl', '/Users/me/repos/bio_ann/imageQA/', 'avg', fix=False)
    # reformat4vipllava(coord_type='raw', fix=False)
    aggregate(coord_type='raw', fix=True)
