import numbers
import os.path as osp

import mmcv
from mmcv.runner import auto_fp16

from ..builder import build_backbone, build_component, build_loss
from ..common import set_requires_grad
from ..registry import MODELS
from .basic_restorer import BasicRestorer

from mmedit.ops import resize

from mmedit.core import tensor2img

@MODELS.register_module()
class SRSWIN(BasicRestorer):
    """SRSWIN model for single image super-resolution.

    Args:
        encoder (dict): Config for the generator.
        decoder (dict): Config for the discriminator.
        pixel_loss (dict): Config for the pixel loss. Default: None.
        perceptual_loss (dict): Config for the perceptual loss. Default: None.
        train_cfg (dict): Config for training. Default: None.
            You may change the training of gan by setting:
            `disc_steps`: how many discriminator updates after one generate
            update;
            `disc_init_steps`: how many discriminator updates at the start of
            the training.
            These two keys are useful when training with WGAN.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 encoder,
                 decoder,
                 upsampler,
                 scale=2,
                 pixel_loss=None,
                 perceptual_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 encoder_pretrained=None,
                 decoder_pretrained=None,
                 upsampler_pretrained=None,
                 align_corners=False):
        super(BasicRestorer, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # encoder
        self.encoder = build_backbone(encoder)
        # decoder
        self.decoder = build_backbone(decoder)
        # upsampler
        self.upsampler = build_backbone(upsampler)

        self.scale = scale
        self.align_corners=align_corners

        # support fp16
        self.fp16_enabled = False

        # loss
        self.pixel_loss = build_loss(pixel_loss) if pixel_loss else None
        self.perceptual_loss = build_loss(
            perceptual_loss) if perceptual_loss else None

        self.disc_steps = 1 if self.train_cfg is None else self.train_cfg.get(
            'disc_steps', 1)
        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))
        self.step_counter = 0  # counting training steps

        self.init_weights(encoder_pretrained, decoder_pretrained, upsampler_pretrained)

    def init_weights(self, encoder_pretrained=None, decoder_pretrained=None, upsampler_pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        self.encoder.init_weights(pretrained=encoder_pretrained)
        self.decoder.init_weights(pretrained=decoder_pretrained)
        self.upsampler.init_weights(pretrained=upsampler_pretrained)

    @auto_fp16(apply_to=('lq', ))
    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        """Forward function.

        Args:
            lq (Tensor): Input lq images.
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """
        if test_mode:
            return self.forward_test(lq, gt, **kwargs)

        raise ValueError(
            'SRGAN model does not supprot `forward_train` function.')

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        # data
        lq = data_batch['lq']
        gt = data_batch['gt']

        # forward - encoder and decoder
        output = self.encoder(lq)
        output = self.decoder(output)

        # simple upsampling
        shape = [i * self.scale for i in list(lq.shape[2:])]
        output = resize(
            input=output,
            size=shape,
            mode='bilinear',
            align_corners=self.align_corners)

        # forward - final conv
        output = self.upsampler(output)

        losses = dict()
        log_vars = dict()

        # calculate loss
        if self.pixel_loss:
            losses['loss_pix'] = self.pixel_loss(output, gt)
        if self.perceptual_loss:
            loss_percep, loss_style = self.perceptual_loss(
                output, gt)
        if loss_percep is not None:
            losses['loss_perceptual'] = loss_percep
        if loss_style is not None:
            losses['loss_style'] = loss_style

        # parse loss
        loss_f, log_vars_f = self.parse_losses(losses)
        log_vars.update(log_vars_f)

        # optimize
        optimizer['encoder'].zero_grad()
        optimizer['decoder'].zero_grad()
        optimizer['upsampler'].zero_grad()
        loss_f.backward()
        optimizer['encoder'].step()
        optimizer['decoder'].step()
        optimizer['upsampler'].step()

        self.step_counter += 1

        log_vars.pop('loss')  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))

        return outputs

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        # forward - encoder and decoder
        output = self.encoder(lq)
        output = self.decoder(output)

        # simple upsampling
        shape = [i * self.scale for i in list(lq.shape[2:])]
        output = resize(
            input=output,
            size=shape,
            mode='bilinear',
            align_corners=self.align_corners)

        # forward - final conv
        output = self.upsampler(output)

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(output), save_path)

        return results