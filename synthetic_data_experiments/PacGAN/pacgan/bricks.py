"""ALI-related bricks."""
from theano import tensor
from blocks.bricks.base import Brick, application, lazy
from blocks.bricks.conv import ConvolutionalSequence
from blocks.bricks.interfaces import Initializable, Random
from blocks.select import Selector


class PacGAN(Initializable, Random):
    """Packed generative adversarial networks.

    Parameters
    ----------
    decoder : :class:`blocks.bricks.Brick`
        Decoder network.
    discriminator : :class:`blocks.bricks.Brick`
        Discriminator network.

    """
    def __init__(self, decoder, discriminator, **kwargs):
        self.decoder = decoder
        self.discriminator = discriminator

        super(PacGAN, self).__init__(**kwargs)
        self.children.extend([self.decoder, self.discriminator])

    @property
    def discriminator_parameters(self):
        return list(
            Selector([self.discriminator]).get_parameters().values())

    @property
    def generator_parameters(self):
        return list(
            Selector([self.decoder]).get_parameters().values())

    @application(inputs=['z'], outputs=['x_tilde'])
    def sample_x_tilde(self, z, application_call):
        x_tilde = self.decoder.apply(z)

        return x_tilde

    @application(inputs=['x', 'x_tilde'],
                 outputs=['data_preds', 'sample_preds'])
    def get_predictions(self, x, x_tilde, application_call):
        data_sample_preds = self.discriminator.apply(
            tensor.unbroadcast(tensor.concatenate([x, x_tilde], axis=0),
                               *range(x.ndim)))
        data_preds = data_sample_preds[:x.shape[0]]
        sample_preds = data_sample_preds[x.shape[0]:]

        application_call.add_auxiliary_variable(
            tensor.nnet.sigmoid(data_preds).mean(), name='data_accuracy')
        application_call.add_auxiliary_variable(
            (1 - tensor.nnet.sigmoid(sample_preds)).mean(),
            name='sample_accuracy')

        return data_preds, sample_preds

    @application(inputs=['x'],
                 outputs=['discriminator_loss', 'generator_loss'])
    def compute_losses(self, x, zs, application_call):
        x_tildes = []
        for z in zs:
            x_tilde = self.sample_x_tilde(z)
            x_tildes.append(x_tilde)
        data_preds, sample_preds = self.get_predictions(x, tensor.concatenate(x_tildes, axis=1))

        discriminator_loss = (tensor.nnet.softplus(-data_preds) +
                              tensor.nnet.softplus(sample_preds)).mean()
        generator_loss = (tensor.nnet.softplus(data_preds) +
                          tensor.nnet.softplus(-sample_preds)).mean()

        return discriminator_loss, generator_loss

    @application(inputs=['z'], outputs=['samples'])
    def sample(self, z):
        return self.sample_x_tilde(z)