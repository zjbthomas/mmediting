from mmcv.runner import auto_fp16

from ..builder import build_backbone, build_component, build_loss
from ..common import set_requires_grad
from ..registry import MODELS
from .basic_restorer import BasicRestorer


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
                 scale=2,
                 pixel_loss=None,
                 perceptual_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 encoder_pretrained=None,
                 decoder_pretrained=None):
        super(BasicRestorer, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # encoder
        self.encoder = build_backbone(encoder)
        # decoder
        self.decoder = build_backbone(decoder)

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

        self.init_weights(encoder_pretrained, decoder_pretrained)

    def init_weights(self, encoder_pretrained=None, decoder_pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        self.encoder.init_weights(pretrained=encoder_pretrained)
        self.decoder.init_weights(pretrained=decoder_pretrained)

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

        # forward - encoder
        output = self.encoder(lq)

        # notify the size to decoder
        self.decoder.set_output_size(lq.shape[2:] * self.scale)

        # forward - decoder
        output = self.decoder(output)

        print(output.size())
        print(gt.size())

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
        loss_f.backward()
        optimizer['encoder'].step()
        optimizer['decoder'].step()

        self.step_counter += 1

        log_vars.pop('loss')  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))

        return outputs
