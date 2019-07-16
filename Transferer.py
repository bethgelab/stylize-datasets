import net
from function import adaptive_instance_normalization, input_transform
import torch
import torch.nn as nn


class StyleTransfer():
    def __init__(self, alpha, content_size, style_size, crop):
        self.vgg = net.vgg
        self.decoder = net.decoder

        assert (0.0 <= alpha <= 1.0)
        self.alpha = alpha

        self.content_size = content_size
        self.style_size = style_size
        self.crop = crop

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.decoder.eval()
        self.vgg.eval()

        self.decoder.load_state_dict(torch.load('models/decoder.pth'))
        self.vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])

        self.vgg.to(self.device)
        self.decoder.to(self.device)

        self.content_tf = input_transform(self.content_size, self.crop)
        self.style_tf = input_transform(self.style_size, self.crop)

    def stylize(self, content, style):
        content = self.content_tf(content)
        content = content.to(self.device).unsqueeze(0)
        style = self.style_tf(style)
        style = style.to(self.device).unsqueeze(0)

        with torch.no_grad():
            content_f = self.vgg(content)
            style_f = self.vgg(style)
            feat = adaptive_instance_normalization(content_f, style_f)
            feat = feat * self.alpha + content_f * (1 - self.alpha)
            output = self.decoder(feat)
        return output.cpu()
