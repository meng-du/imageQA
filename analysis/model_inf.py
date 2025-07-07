import pickle
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, VipLlavaForConditionalGeneration


IMG_DIR = 'images/'
PROMPT = 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\'s questions.###Human: <image>\n{}###Assistant:'


def vipllava_inference(save_attn=True):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    torch.manual_seed(42)

    with torch.no_grad():
        processor = AutoProcessor.from_pretrained('llava-hf/vip-llava-7b-hf')
        model = VipLlavaForConditionalGeneration.from_pretrained('llava-hf/vip-llava-7b-hf', torch_dtype=torch.float16).to(device)

        data = pickle.load(open('questions.pkl', 'rb'))

        output = {}
        for i, img in enumerate(sorted(list(data.keys()))):
            print(f'Processing {i}, {img}...')
            image = Image.open(IMG_DIR + img.replace('.png', '.jpg')).convert('RGB')
            attns, output[img] = {}, {}
            for block in tqdm(data[img].keys()):
                question = data[img][block]['question']
                prompt = PROMPT.format(question)

                inputs = processor(text=prompt, images=image, return_tensors='pt').to(device, torch.float16)

                if save_attn:
                    output = model(**inputs, output_attentions=True, output_hidden_states=True)
                    attns[block] = np.squeeze([t.cpu().detach().numpy() for t in output.attentions])
                else:
                    generate_ids = model.generate(**inputs, max_new_tokens=200)
                    output_text = processor.decode(generate_ids[0][len(inputs['input_ids'][0]):])
                    output[img][block] = {
                        'question': question,
                        'input_ids': inputs['input_ids'][0].cpu().numpy().tolist(),
                        'answer': output_text
                    }

                torch.cuda.empty_cache()

            if save_attn:
                np.savez_compressed(f'vipllava_attn_{img[:-4]}.npz', **attns)
        if not save_attn:
            with open(f'vipllava_answers.pkl', 'wb') as f:
                pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)

        # attention: 32, 32, 626, 626 layers, heads, tokens, tokens


if __name__ == '__main__':
    vipllava_inference(save_attn=False)
