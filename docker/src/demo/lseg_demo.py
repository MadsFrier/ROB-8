import os
import math
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import clip
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt
import gc

# add directories to path for code to find it
sys.path.insert(0, '/workspaces/ROB-8/docker/src/content/vlmaps/utils')
sys.path.insert(0, '/workspaces/ROB-8/docker/src/content/vlmaps/lseg/modules')
sys.path.insert(0, '/workspaces/ROB-8/docker/src/content/vlmaps/lseg/modules/models')
sys.path.insert(0, '/workspaces/ROB-8/docker/src/content/vlmaps/lseg/additional_utils')

from clip_mapping_utils import get_new_pallete, get_new_mask_pallete

from lseg_net import LSegEncNet
from models import resize_image, pad_image, crop_image, LSeg_MultiEvalModule

def load_lseg(prompt, crop_size=480, base_size=520):
    with torch.no_grad():
        labels = prompt.split(",")

        # loading models
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device: " + device)
        clip_version = "ViT-B/32"
        print("Loading CLIP model...")
        clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
        clip_model.to(device).eval()
        lang_token = clip.tokenize(labels)
        lang_token = lang_token.to(device)
        
        with torch.no_grad():
            text_feats = clip_model.encode_text(lang_token)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        text_feats = text_feats.cpu().numpy()
        model = LSegEncNet(prompt, arch_option=0,
                            block_depth=0,
                            activation='lrelu',
                            crop_size=crop_size,
                            base_size=base_size,)
        model_state_dict = model.state_dict()
        
        pretrained_state_dict = torch.load("/workspaces/ROB-8/docker/src/content/vlmaps/lseg/checkpoints/demo_e200.ckpt")

        pretrained_state_dict = {k.lstrip('net.'): v for k, v in pretrained_state_dict['state_dict'].items()}
        model_state_dict.update(pretrained_state_dict)
        model.load_state_dict(pretrained_state_dict)
        
        model.eval()
        model = model.cuda()
        
        model._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        
        scales = ([0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]) 

        norm_mean= [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]
        
        model.mean = norm_mean
        model.std = norm_std
        
        evaluator = LSeg_MultiEvalModule(model, scales=scales, flip=True).cuda()
        evaluator.eval()
    
    return model, labels

def run_lseg(image_np, model, labels, palette, show):
    with torch.no_grad():
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        image = transform(image_np).unsqueeze(0)
        
        with torch.no_grad():
            outputs, predict = get_lseg_feat(model, image_np, labels, transform)
            mask, patches = get_new_mask_pallete(predict, palette, out_label_flag=True, labels=labels)
        
        if show:
            new_palette = get_new_pallete(len(labels))
            img = image[0].permute(1,2,0)
            img = img * 0.5 + 0.5
            img = Image.fromarray(np.uint8(255*img)).convert("RGBA")
            seg = mask.convert("RGBA")
            f, axarr = plt.subplots(1,2)
            axarr[0].imshow(img)
            plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.5, 1), prop={'size': 20})
            axarr[1].imshow(seg)
            plt.show()
    
    return predict, mask.convert('RGB'), patches

def get_lseg_feat(model: LSegEncNet, image: np.array, labels, transform, crop_size=480, \
                 base_size=520, norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5]):
    with torch.no_grad():
        #vis_image = image.copy()
        image = transform(image).unsqueeze(0).cuda()
        img = image[0].permute(1,2,0)
        img = img * 0.5 + 0.5

        batch, _, h, w = image.size()
        stride_rate = 2.0/3.0
        stride = int(crop_size * stride_rate)

        long_size = base_size
        if h > w:
            height = long_size
            width = int(1.0 * w * long_size / h + 0.5)
            short_size = width
        else:
            width = long_size
            height = int(1.0 * h * long_size / w + 0.5)
            short_size = height


        cur_img = resize_image(image, height, width, **{'mode': 'bilinear', 'align_corners': True})

        if long_size <= crop_size:
            pad_img = pad_image(cur_img, norm_mean,
                                norm_std, crop_size)
            print(pad_img.shape)
            with torch.no_grad():
                outputs, logits = model(pad_img, labels)
            outputs = crop_image(outputs, 0, height, 0, width)
        else:
            if short_size < crop_size:
                # pad if needed
                pad_img = pad_image(cur_img, norm_mean,
                                    norm_std, crop_size)
            else:
                pad_img = cur_img
            _,_,ph,pw = pad_img.shape #.size()
            assert(ph >= height and pw >= width)
            h_grids = int(math.ceil(1.0 * (ph-crop_size)/stride)) + 1
            w_grids = int(math.ceil(1.0 * (pw-crop_size)/stride)) + 1
            with torch.cuda.device_of(image):
                with torch.no_grad():
                    outputs = image.new().resize_(batch, model.out_c,ph,pw).zero_().cuda()
                    logits_outputs = image.new().resize_(batch, len(labels),ph,pw).zero_().cuda()
                count_norm = image.new().resize_(batch,1,ph,pw).zero_().cuda()
            # grid evaluation
            for idh in range(h_grids):
                for idw in range(w_grids):
                    h0 = idh * stride
                    w0 = idw * stride
                    h1 = min(h0 + crop_size, ph)
                    w1 = min(w0 + crop_size, pw)
                    crop_img = crop_image(pad_img, h0, h1, w0, w1)
                    # pad if needed
                    pad_crop_img = pad_image(crop_img, norm_mean,
                                                norm_std, crop_size)
                    with torch.no_grad():
                        output, logits = model(pad_crop_img, labels)
                    cropped = crop_image(output, 0, h1-h0, 0, w1-w0)
                    cropped_logits = crop_image(logits, 0, h1-h0, 0, w1-w0)
                    outputs[:,:,h0:h1,w0:w1] += cropped
                    logits_outputs[:,:,h0:h1,w0:w1] += cropped_logits
                    count_norm[:,:,h0:h1,w0:w1] += 1
            assert((count_norm==0).sum()==0)
            outputs = outputs / count_norm
            logits_outputs = logits_outputs / count_norm
            outputs = outputs[:,:,:height,:width]
            logits_outputs = logits_outputs[:,:,:height,:width]
        outputs = outputs.cpu()
        outputs = outputs.numpy() # B, D, H, W
        predicts = [torch.max(logit, 0)[1].cpu().numpy() for logit in logits_outputs]
        predict = predicts[0]

    return outputs, predict

def torch_clear_cache(print_cache):
    with torch.no_grad():
        if print_cache:
            print('Before clearing: ', torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
        gc.collect()
        torch.cuda.empty_cache()
        if print_cache:
            print('After clearing: ', torch.cuda.memory_allocated(), torch.cuda.memory_reserved())

if __name__ == "__main__":
    
    # choose directory of images to load in
    data_dir = "/workspaces/ROB-8/docker/src/content/demo_images/" # USE THIS FOR DEMO

    # choose image to load in
    img_name = "test_plant.jpg" # USE THIS FOR DEMO

    # choose prompt
    prompt = 'tree, animal, sky, grass, other' # USE THIS FOR DEMO
    prompt = 'other, chair, lamp, door, wall, carpet, floor, shelf, table'
    prompt = 'other, package, door, brick, floor, carpet'
    prompt = 'other, plant, table, wall'
    # choose whether to show the image
    show = True
    
    # segment image using lseg
    model, labels = load_lseg(prompt)
    
    img = cv2.cvtColor(cv2.imread(data_dir + img_name), cv2.COLOR_BGR2RGB)
    
    palette = new_palette = get_new_pallete(len(labels))
    
    run_lseg(img, model, labels, palette, show)