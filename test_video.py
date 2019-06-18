import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from model import Model

import cv2
import time


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.ToTensor(),
                            normalize])


def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


def main():
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer by Pytorch')
    parser.add_argument('--style', '-s', type=str, default=None,
                        help='Style image path e.g. image.jpg')
    parser.add_argument('--alpha', '-a', type=int, default=1,
                        help='alpha control the fusion degree in Adain')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(nagative value indicate CPU)')
    parser.add_argument('--model_state_path', type=str, default='model_state.pth',
                        help='save directory for result and loss')

    args = parser.parse_args()

    # set device on GPU if available, else CPU
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    # set model
    model = Model()
    if args.model_state_path is not None:
        model.load_state_dict(torch.load(args.model_state_path, map_location=lambda storage, loc: storage))
    model = model.to(device)

    # style image
    s_img = Image.open(args.style)
    s_tensor = trans(s_img).unsqueeze(0).to(device)

    # camera
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        c_img = Image.fromarray(frame[:, :, ::-1])
        c_tensor = trans(c_img).unsqueeze(0).to(device)

        current_time = time.time()
        with torch.no_grad():
            out = model.generate(c_tensor, s_tensor, args.alpha)
        finished_time = time.time()
        print('fps: {}'.format(1 / (finished_time - current_time)))

        out_denorm = denorm(out, device)
        out_denorm_np = out_denorm.squeeze().cpu().numpy().transpose(1, 2, 0)
        out_denorm_np = out_denorm_np[:, :, ::-1]
        cv2.imshow('Result', out_denorm_np)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
