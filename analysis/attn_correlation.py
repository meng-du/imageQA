import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
from scipy.stats import pearsonr


N_LAYERS = 32
blocks = ['fact-ppl', 'fact-nonppl', 'inference-social', 'inference-nonsocial', 'next', 'relation', 'feel']


def get_img2txt_attn(answers):
    img2txt_attn = {}
    for img in tqdm(answers.keys()):
        imname = img[:-4]
        imattn = np.load(f'attns/vipllava_attn_{imname}.npz', allow_pickle=True)
        img2txt_attn[imname] = {}
        for block in blocks:
            img_idx = np.where(np.array(answers[img][block]['input_ids']) == 32000)  # 32000 is the <image> token
            txt_start = np.where(np.array(answers[img][block]['input_ids']) == 13)[0][0] + 1
            txt_end = np.where(np.array(answers[img][block]['input_ids']) == 2277)[0][-1]
            assert 612 < txt_start < txt_end
            txt_idx = slice(txt_start, txt_end)
            i2t_attn = np.squeeze(imattn[block][:, :, txt_idx, img_idx])
            i2t_attn = np.mean(i2t_attn, axis=2)  # average across all question text tokens
            i2t_attn = np.mean(i2t_attn, axis=1)  # average across all heads
            i2t_attn = i2t_attn.reshape((-1, 24, 24))
            assert i2t_attn.shape == (32, 24, 24)
            img2txt_attn[imname][block] = i2t_attn
    return img2txt_attn


def correlate(data1, data2, method='pearson'):
    if data1.dtype == data2.dtype == np.float16:
        data1 = data1.astype(np.float32)
        data2 = data2.astype(np.float32)
    if method == 'pearson':
        return pearsonr(data1.flatten(), data2.flatten()).statistic
    elif method == '2d':
        return correlate2d(data1, data2, mode='valid')[0, 0]
    else:
        raise ValueError('Invalid method')


def correlate_across_data(attn, eyedata, method='pearson'):
    correlations = {}
    for block in blocks:
        correlations[block] = [[] for _ in range(N_LAYERS)]
        for imname in attn.keys():
            if imname not in eyedata:
                print(f'{imname} not in eyedata, skipping...')
                continue
            for layer in range(N_LAYERS):
                layer_attn = attn[imname][block][layer]
                eye_attn = eyedata[imname][block]
                corr = correlate(layer_attn, eye_attn, method=method)
                if not np.isnan(corr):
                    correlations[block][layer].append(corr)
    return correlations


def correlate_across_condition(data, method='pearson'):
    correlations = [[[] for _ in range(len(blocks))] for _ in range(len(blocks))]  # 7x7
    for i, block in enumerate(blocks):
        for j, other_block in enumerate(blocks):
            if len(data['co014800'][block].shape) == 3:
                correlations[i][j] = [[] for _ in range(N_LAYERS)]
            for imname in data.keys():
                if len(imname) == 3:
                    continue
                if len(data[imname][block].shape) == 3:
                    for layer in range(N_LAYERS):
                        attn = data[imname][block][layer]
                        other_attn = data[imname][other_block][layer]
                        corr = correlate(attn, other_attn, method=method)
                        if not np.isnan(corr):
                            correlations[i][j][layer].append(corr)
                elif len(data[imname][block].shape) == 2:
                    attn = data[imname][block]
                    other_attn = data[imname][other_block]
                    corr = correlate(attn, other_attn, method=method)
                    if not np.isnan(corr):
                        correlations[i][j].append(corr)
                else:
                    raise ValueError('Expected 2D or 3D array')
    if len(data[imname][block].shape) == 3:
        return np.mean(correlations, axis=3)
    else:
        correlations = [[np.mean(correlations[i][j]) for i in range(len(blocks))] for j in range(len(blocks))]
        return np.array(correlations)


if __name__ == '__main__':
    # answers = pickle.load(open(f'vipllava_answers.pkl', 'rb'))
    # attn = get_img2txt_attn(answers)
    # with open('vipllava_img2txt_attn.pkl', 'wb') as f:
    #     pickle.dump(attn, f)

    attn = pickle.load(open('/Users/me/repos/bio_ann/imageQA/analysis/vipllava_img2txt_attn.pkl', 'rb'))
    eyedata = pickle.load(open('/Users/me/repos/bio_ann/imageQA/analysis/eyedata_aggregated_raw_fix.pkl', 'rb'))
    # correlations = correlate_across_data(attn, eyedata)
    # correlations = correlate_across_condition(attn, method='pearson')
    # print(correlations.shape)

    for layer in range(N_LAYERS):
        correlations = correlate_across_condition(attn, method='pearson')
        plt.figure(figsize=(5, 5))
        plt.imshow(correlations[:,:,layer], vmax=1, cmap='coolwarm')
        plt.colorbar()
        plt.xticks(np.arange(len(blocks)), blocks, rotation=90)
        plt.yticks(np.arange(len(blocks)), blocks)
        plt.title(f'Layer {layer}')
        plt.savefig(f'vipllava_attn_layer{layer}_correlation.png', dpi=300)
