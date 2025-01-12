import argparse


def get_opts():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dir_name', type=str)
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval', 'trainvaltest'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor (<=1.0) for the images')

    # model parameters
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    parser.add_argument('--stage_num', type=int, default=3,
                        help='stage number')
    parser.add_argument('--sem_num', type=int, default=3,
                        help='sem number')
    parser.add_argument('--share_grid', action='store_true', default=False,
                        help='all stage share one grid')

    # loss parameters
    parser.add_argument('--composite_weight', type=float, default=1)
    parser.add_argument('--semantic_weight', type=float, default=1e-2)
    parser.add_argument('--l1TimePlanes_weight', type=float, default=1e-3)
    parser.add_argument('--timeSmoothness_weight', type=float, default=1e-3)
    parser.add_argument('--distortion_weight', type=float, default=1e-2)
    parser.add_argument('--opacity_weight', type=float, default=1e-3)
    parser.add_argument('--density_weight', type=float, default=1e-3)
    parser.add_argument('--bdc_weight', type=float, default=1e-3)

    # training options
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of rays in a batch')
    parser.add_argument('--ray_sampling_strategy', type=str, default='all_images',
                        choices=['all_images', 'same_image'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        ''')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of training epochs')
    parser.add_argument('--stage_end_epoch', type=int, default=5,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')

    parser.add_argument('--random_bg', action='store_true', default=False,
                        help='''whether to train with random bg color (real scene only)
                        to avoid objects with black color to be predicted as transparent
                        ''')

    # validation options
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')

    # misc
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained checkpoint to load (excluding optimizers, etc)')

    return parser.parse_args()
