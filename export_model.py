import argparse

import torch
import torch.nn as nn

from model import VGGEncoder, Decoder, adain


parser = argparse.ArgumentParser(description='AdaIN Style Transfer by Pytorch')
parser.add_argument('--gpu', '-g', type=int, default=0,
                    help='GPU ID(nagative value indicate CPU)')
parser.add_argument('--model_state_path', type=str, default='model_state.pth',
                    help='save directory for result and loss')
args = parser.parse_args()
print(args)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_encoder = VGGEncoder()
        self.decoder = Decoder()

    def forward(self, content_images, style_images, alpha):
        content_features = self.vgg_encoder(content_images, output_last_feature=True)
        style_features = self.vgg_encoder(style_images, output_last_feature=True)
        t = adain(content_features, style_features)
        t = alpha * t + (1 - alpha) * content_features
        out = self.decoder(t)
        return out


def main():
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    model = Model()
    if args.model_state_path is not None:
        model.load_state_dict(torch.load(args.model_state_path, map_location=lambda storage, loc: storage))
        print('Succeed to load state dict!')
    model.to(device)
    model.eval()

    input_c = torch.ones(1, 3, 1080, 1920).to(device)
    input_s = torch.ones(1, 3, 360, 640).to(device)
    input_a = torch.ones(1).to(device)
    traced_script_module = torch.jit.trace(model, (input_c, input_s, input_a))

    if device == 'cpu':
        name = 'AdaIN_cpu.pt'
    else:
        name = 'AdaIN_gpu.pt'
    traced_script_module.save(name)


if __name__ == "__main__":
    main()
