import sys

sys.path.append("../")
import argparse
import torch
import torch.nn.functional as F
import copy
import warnings
from utils.utils_baseline import get_network, get_eval_pool, evaluate_synset, ParamDiffAug
from load_data import load_comp_dd
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(args):
    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    im_size = (args.res, args.res)
    channel, num_classes, dst_train, dst_test, class_map, class_map_inv = load_comp_dd(
        args.data_path,
        args.category,
        args.subset,
        im_size,
        args.batch_size
    )
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    args.im_size = im_size
    data_save = []

    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    args.distributed = torch.cuda.device_count() > 1

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    def SoftCrossEntropy(inputs, target, reduction='average'):
        input_log_likelihood = -F.log_softmax(inputs, dim=1)
        target_log_likelihood = F.softmax(target, dim=1)
        batch = inputs.shape[0]
        loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch
        return loss

    soft_cri = SoftCrossEntropy

    image_syn_eval = torch.load(args.data_dir)
    label_syn_eval = torch.load(args.label_dir)
    args.lr_net = torch.load(args.lr_dir)

    testloader = DataLoader(dst_test, batch_size=args.batch_size, shuffle=False)

    for model_eval in model_eval_pool:
        print('Evaluating: ' + model_eval)
        network = get_network(model_eval, channel, num_classes, im_size, dist=False).to(args.device)
        _, acc_train, acc_test = evaluate_synset(0, copy.deepcopy(network), image_syn_eval, label_syn_eval, testloader,
                                                 args, texture=False, train_criterion=soft_cri)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--category', type=str, default='bird', help='category')
    parser.add_argument('--subset', type=str, default='easy')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')
    parser.add_argument('--epoch_eval_train', type=int, default=1000,
                        help='epochs to train a model with synthetic data')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for real data')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--lr_net', type=float, default=0.01, help='initialization for synthetic learning rate')
    parser.add_argument('--data_dir', type=str, default='path', help='dataset')
    parser.add_argument('--label_dir', type=str, default='path', help='dataset')
    parser.add_argument('--lr_dir', type=str, default='path', help='dataset')

    args = parser.parse_args()
    main(args)
