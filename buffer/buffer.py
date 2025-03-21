import os
import argparse
import sys
import time
sys.path.append("../")
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.utils_gsam import get_dataset, get_network, get_daparam, get_imagenet_1k, \
    TensorDataset, epoch, ParamDiffAug
import copy
import multiprocessing as mp
import warnings
import torchvision.transforms as transforms
warnings.filterwarnings("ignore", category=DeprecationWarning)


def train(args, channel, num_classes, im_size, trainloader, testloader, save_dir):

    trajectories = []
    for it in range(args.num_experts):
        ''' Train synthetic data '''
        if hasattr(args, 'im_size') and args.im_size > 0:
            model_im_size = (args.im_size, args.im_size)
            print(f"使用命令行参数中的图像大小: {model_im_size}")
        else:
            model_im_size = im_size
            
        teacher_net = get_network(args.model, channel, num_classes, model_im_size).to(args.device) # get a random model
        teacher_net.train()
        lr = args.lr_teacher
        
        criterion = nn.CrossEntropyLoss().to(args.device)
        
        ##modification: using FTD here 
        from gsam import GSAM, LinearScheduler, CosineScheduler, ProportionScheduler
        base_optimizer = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
        # scheduler = CosineScheduler(T_max=args.train_epochs*len_dst_train, max_value=lr, min_value=0.0, 
            # optimizer=base_optimizer)
        scheduler = torch.optim.lr_scheduler.StepLR(base_optimizer,step_size=args.train_epochs*len(trainloader),gamma=1)
        rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=lr, min_lr=lr,
            max_value=args.rho_max, min_value=args.rho_min)
        teacher_optim = GSAM(params=teacher_net.parameters(), base_optimizer=base_optimizer, 
                model=teacher_net, gsam_alpha=args.alpha, rho_scheduler=rho_scheduler, adaptive=args.adaptive)


        teacher_optim.zero_grad()

        timestamps = []

        timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

        lr_schedule = [args.train_epochs // 2 + 1]
        for e in range(args.train_epochs):

            train_loss, train_acc = epoch("train", dataloader=trainloader, net=teacher_net, optimizer=teacher_optim,
                                        criterion=criterion, args=args, aug=True,scheduler=scheduler)

            test_loss, test_acc = epoch("test", dataloader=testloader, net=teacher_net, optimizer=None,
                                        criterion=criterion, args=args, aug=False, scheduler=scheduler)

            print("Itr: {}\tEpoch: {}\tTrain Acc: {}\tTest Acc: {}".format(it, e, train_acc, test_acc))

            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])
            
        trajectories.append(timestamps)
        if len(trajectories) == args.save_interval:
            n = 0
            while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
                n += 1
            print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
            trajectories = []


def main(args):

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()

    if args.dataset != "ImageNet":
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, \
        class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
        
        # 使用命令行参数的num_classes覆盖默认值（如果存在）
        if hasattr(args, 'num_classes') and args.num_classes > 0:
            num_classes = args.num_classes
            print(f"覆盖类别数为命令行参数值: {num_classes}")
    
    else:
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, \
        class_map, class_map_inv = get_imagenet_1k(args, args.data_path)

    # print('\n================== Exp %d ==================\n '%exp)
    print('Hyper-parameters: \n', args.__dict__)

    save_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        save_dir = os.path.join(save_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        save_dir += "_NO_ZCA"
    save_dir = os.path.join(save_dir, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    
    # 添加固定大小变换
    resize_transform = transforms.Compose([
        transforms.Resize((256, 256))  # 调整所有图像为相同大小
    ])
    
    print("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        img = sample[0]
        # 确保图像大小一致
        img = resize_transform(img)
        images_all.append(torch.unsqueeze(img, dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])
    #print('num of training images',len(images_all))
    len_dst_train = len(images_all)  ##50000

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    dst_train = TensorDataset(copy.deepcopy(images_all.detach()), copy.deepcopy(labels_all.detach()))
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    ''' set augmentation for whole-dataset training '''
    args.dc_aug_param = get_daparam(args.dataset, args.model, args.model, None)
    args.dc_aug_param['strategy'] = 'crop_scale_rotate'  # for whole-dataset training
    print('DC augmentation parameters: \n', args.dc_aug_param)

    train(args, channel, num_classes, im_size, trainloader, testloader, save_dir)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='subset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=0, help='number of classes, 0 means use default')
    parser.add_argument('--im_size', type=int, default=0, help='size for both height and width of the input image')
    #parser.add_argument('--rho', type=float, default=0.05)
    parser.add_argument("--rho_max", default=2.0, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--rho_min", default=2.0, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--alpha", default=0.4, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    args = parser.parse_args()
    
    mp.set_start_method('spawn')
    main(args)

    


