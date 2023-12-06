import argparse
import os
import random
import numpy as np
import torch
from networks.model import VRBModel
from networks.traj import TrajAffCVAE
from inference import run_inference
from PIL import Image
from pathlib import Path

def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.manual_seed)
    else:
        device = torch.device("cpu")
        torch.manual_seed(args.manual_seed)
    print(f"Using device: {device}")
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    hand_head = TrajAffCVAE(in_dim=2*args.traj_len, hidden_dim=args.hidden_dim,
                        latent_dim=args.hand_latent_dim, condition_dim=args.cond_dim,
                        coord_dim=args.coord_dim, traj_len=args.traj_len)
        
    #resnet output
    if args.resnet_type == 'resnet50': 
        src_in_features = 2048
    else: 
        src_in_features = 512

    net = VRBModel(src_in_features=src_in_features,
                            num_patches=1,
                            hidden_dim=args.hidden_dim, 
                            hand_head=hand_head,
                            encoder_time_embed_type=args.encoder_time_embed_type,
                            num_frames_input=10,
                            resnet_type=args.resnet_type, 
                            embed_dim=args.cond_dim, coord_dim=args.coord_dim,
                            num_heads=args.num_heads, enc_depth=args.enc_depth, 
                            attn_kp=args.attn_kp, attn_kp_fc=args.attn_kp_fc, n_maps=5)

    dt = torch.load(args.model_path, map_location=device)
    net.load_state_dict(dt)
    net = net.to(device)
    image_pil = Image.open(args.image).convert("RGB")
    # image_pil = image_pil.resize((1008, 756))
    object_list = []
    with open(args.obj_list, 'r') as f:
        for line in f.readlines():
            object_list.append(line.strip())
    print(object_list)
    im_out = run_inference(net, image_pil, object_list, args.overlap, args.max_box, device=device)
    if args.output is None:
        args.output = os.path.splitext(args.image)[0] + '_out.png'
        # output_path = Path(args.output)
    im_out.save(args.output) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_heads', type=int, default=8, help='num of heads in transformer')
    parser.add_argument('--enc_depth', type=int, default=6, help='transformer encoder depth')
    parser.add_argument('--hidden_dim', type=int, default=192, help="hidden feature dimension")
    parser.add_argument('--hand_latent_dim', type=int, default=4, help="Latent dimension for trajectory CVAE")
    parser.add_argument('--cond_dim', type=int, default=256, help="downprojection dimension for transformer encoder")
    parser.add_argument('--coord_dim', type=int, default=64, help='Contact coordinate feature dimension')
    parser.add_argument('--resnet_type', type=str, default='resnet18')
    parser.add_argument('--attn_kp', type=int, default=1)
    parser.add_argument('--attn_kp_fc', type=int, default=1)
    parser.add_argument('--traj_len', type=int, default=5)
    parser.add_argument("--encoder_time_embed_type", default="sin",  choices=["sin", "param"], help="transformer encoder time position embedding")
    parser.add_argument("--manual_seed", default=0, type=int, help="manual seed")
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--output', type=str)
    parser.add_argument('--obj_list', type=str, required=True)
    parser.add_argument('--overlap', type=int, default=0.5)
    parser.add_argument('--max_box', type=int, required=True)
    parser.add_argument('--model_path', type=str, default='./models/model_checkpoint_1249.pth.tar')
    args = parser.parse_args()
    

    main(args)
    print("All done !")