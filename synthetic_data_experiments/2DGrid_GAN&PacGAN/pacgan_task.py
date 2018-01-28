from gpu_task_scheduler.gpu_task import GPUTask
from pacgan.bricks import PacGAN
from pacgan.distributions import circle_gaussian_mixture
from pacgan.streams import create_packing_gaussian_mixture_data_streams
from pacgan.extensions import ModelLogger, GraphLogger, MetricLogger

from ali.algorithms import ali_algorithm
from ali.utils import as_array

import numpy, os, random, fuel, blocks
from theano import tensor, function
from blocks.algorithms import Adam
from blocks.bricks import MLP, Rectifier, Identity, LinearMaxout, Linear
from blocks.bricks.bn import BatchNormalization
from blocks.bricks.sequences import Sequence
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.graph import ComputationGraph, apply_dropout
from blocks.graph.bn import (batch_normalization, get_batch_normalization_updates)
from blocks.filter import VariableFilter
from blocks.initialization import IsotropicGaussian, Constant
from blocks.model import Model
from blocks.main_loop import MainLoop
from blocks.roles import INPUT
from blocks.select import Selector


class PacGANTask(GPUTask):
    def required_env(self):
        ans = {}
        if self._config["blocks_random"]:
            seed = random.randint(1, 100000)
            blocksrc_path = os.path.join(self._work_dir, ".blocksrc")
            f = open(blocksrc_path, "w")
            f.write("default_seed: {}\n".format(seed))
            f.close()
            blocks.config.config.default_seed = seed
            ans = {"BLOCKS_CONFIG": blocksrc_path}
        return ans
        
    def create_model_brick(self):
        decoder = MLP(
            dims=[self._config["num_zdim"], self._config["gen_hidden_size"], self._config["gen_hidden_size"], self._config["gen_hidden_size"], self._config["gen_hidden_size"], self._config["num_xdim"]],
            activations=[Sequence([BatchNormalization(self._config["gen_hidden_size"]).apply,
                                   self._config["gen_activation"]().apply],
                                  name='decoder_h1'),
                         Sequence([BatchNormalization(self._config["gen_hidden_size"]).apply,
                                   self._config["gen_activation"]().apply],
                                  name='decoder_h2'),
                         Sequence([BatchNormalization(self._config["gen_hidden_size"]).apply,
                                   self._config["gen_activation"]().apply],
                                  name='decoder_h3'),
                         Sequence([BatchNormalization(self._config["gen_hidden_size"]).apply,
                                   self._config["gen_activation"]().apply],
                                  name='decoder_h4'),
                         Identity(name='decoder_out')],
            use_bias=False,
            name='decoder')

        discriminator = Sequence(
            application_methods=[
                LinearMaxout(
                    input_dim=self._config["num_xdim"] * self._config["num_packing"],
                    output_dim=self._config["disc_hidden_size"],
                    num_pieces=self._config["disc_maxout_pieces"],
                    weights_init=IsotropicGaussian(self._config["weights_init_std"]),
                    biases_init=self._config["biases_init"],
                    name='discriminator_h1').apply,
                LinearMaxout(
                    input_dim=self._config["disc_hidden_size"],
                    output_dim=self._config["disc_hidden_size"],
                    num_pieces=self._config["disc_maxout_pieces"],
                    weights_init=IsotropicGaussian(self._config["weights_init_std"]),
                    biases_init=self._config["biases_init"],
                    name='discriminator_h2').apply,
                LinearMaxout(
                    input_dim=self._config["disc_hidden_size"],
                    output_dim=self._config["disc_hidden_size"],
                    num_pieces=self._config["disc_maxout_pieces"],
                    weights_init=IsotropicGaussian(self._config["weights_init_std"]),
                    biases_init=self._config["biases_init"],
                    name='discriminator_h3').apply,
                Linear(
                    input_dim=self._config["disc_hidden_size"],
                    output_dim=1,
                    weights_init=IsotropicGaussian(self._config["weights_init_std"]),
                    biases_init=self._config["biases_init"],
                    name='discriminator_out').apply],
            name='discriminator')

        gan = PacGAN(decoder=decoder, discriminator=discriminator, weights_init=IsotropicGaussian(self._config["weights_init_std"]), biases_init=self._config["biases_init"], name='gan')
        gan.push_allocation_config()
        decoder.linear_transformations[-1].use_bias = True
        gan.initialize()
            
        print("Number of parameters in discriminator: {}".format(numpy.sum([numpy.prod(v.shape.eval()) for v in Selector(gan.discriminator).get_parameters().values()])))
        print("Number of parameters in decoder: {}".format(numpy.sum([numpy.prod(v.shape.eval()) for v in Selector(gan.decoder).get_parameters().values()])))
        
        return gan
        
    def create_models(self):
        gan = self.create_model_brick()
        x = tensor.matrix('features')
        zs = []
        for i in range(self._config["num_packing"]):
            z = circle_gaussian_mixture(num_modes=self._config["num_zmode"], num_samples=x.shape[0], dimension=self._config["num_zdim"], r=self._config["z_mode_r"], std=self._config["z_mode_std"])
            zs.append(z)

        def _create_model(with_dropout):
            cg = ComputationGraph(gan.compute_losses(x, zs))
            if with_dropout:
                inputs = VariableFilter(
                    bricks=gan.discriminator.children[1:],
                    roles=[INPUT])(cg.variables)
                cg = apply_dropout(cg, inputs, 0.5)
                inputs = VariableFilter(
                    bricks=[gan.discriminator],
                    roles=[INPUT])(cg.variables)
                cg = apply_dropout(cg, inputs, 0.2)
            return Model(cg.outputs)

        model = _create_model(with_dropout=False)
        with batch_normalization(gan):
            bn_model = _create_model(with_dropout=False)

        pop_updates = list(set(get_batch_normalization_updates(bn_model, allow_duplicates=True)))
            
        # merge same variables
        names = []
        counts = []
        pop_update_merges = []
        pop_update_merges_finals = []
        for pop_update in pop_updates:
            b = False
            for i in range(len(names)):
                if (pop_update[0].auto_name == names[i]):
                    counts[i] += 1
                    pop_update_merges[i][1] += pop_update[1]
                    b = True
                    break
            if not b:
                names.append(pop_update[0].auto_name)
                counts.append(1)
                pop_update_merges.append([pop_update[0], pop_update[1]])
        for i in range(len(pop_update_merges)):
            pop_update_merges_finals.append((pop_update_merges[i][0], pop_update_merges[i][1] / counts[i]))
        
        bn_updates = [(p, m * 0.05 + p * 0.95) for p, m in pop_update_merges_finals]

        return model, bn_model, bn_updates
        
    def create_main_loop(self):
        model, bn_model, bn_updates = self.create_models()
        gan, = bn_model.top_bricks
        discriminator_loss, generator_loss = bn_model.outputs
        step_rule = Adam(learning_rate=self._config["learning_rate"], beta1=self._config["beta1"])
        algorithm = ali_algorithm(discriminator_loss, gan.discriminator_parameters, step_rule, generator_loss, gan.generator_parameters, step_rule)
        algorithm.add_updates(bn_updates)
        streams = create_packing_gaussian_mixture_data_streams(
            num_packings=self._config["num_packing"], 
            batch_size=self._config["batch_size"], 
            monitoring_batch_size=self._config["monitoring_batch_size"], 
            means=self._config["x_mode_means"], 
            variances=self._config["x_mode_variances"], 
            priors=self._config["x_mode_priors"],
            num_examples=self._config["num_sample"])
        main_loop_stream, train_monitor_stream, valid_monitor_stream = streams
        bn_monitored_variables = (
            [v for v in bn_model.auxiliary_variables if 'norm' not in v.name] +
            bn_model.outputs)
        monitored_variables = (
            [v for v in model.auxiliary_variables if 'norm' not in v.name] +
            model.outputs)
        extensions = [
            Timing(),
            FinishAfter(after_n_epochs=self._config["num_epoch"]),
            DataStreamMonitoring(
                bn_monitored_variables, train_monitor_stream, prefix="train",
                updates=bn_updates),
            DataStreamMonitoring(
                monitored_variables, valid_monitor_stream, prefix="valid"),
            Checkpoint(os.path.join(self._work_dir, self._config["main_loop_file"]), after_epoch=True, after_training=True, use_cpickle=True),
            ProgressBar(),
            Printing(),
        ]
        if self._config["log_models"]:
            extensions.append(ModelLogger(folder=self._work_dir, after_epoch=True))
        if self._config["log_figures"]:
            extensions.append(GraphLogger(num_modes=self._config["num_zmode"], num_samples=self._config["num_log_figure_sample"], dimension=self._config["num_zdim"], r=self._config["z_mode_r"], std=self._config["z_mode_std"], folder=self._work_dir, after_epoch=True, after_training=True))
        if self._config["log_metrics"]:
            extensions.append(MetricLogger(means=self._config["x_mode_means"], variances=self._config["x_mode_variances"], folder=self._work_dir, after_epoch=True))
        main_loop = MainLoop(model=bn_model, data_stream=main_loop_stream, algorithm=algorithm, extensions=extensions)
        return main_loop
        
    def main(self):
        if self._config["fuel_random"]:
            seed = random.randint(1, 100000)
            fuelrc_path = os.path.join(self._work_dir, ".fuelrc")
            f = open(fuelrc_path, "w")
            f.write("default_seed: {}\n".format(seed))
            f.close()
            fuel.config.default_seed = seed

        main_loop = self.create_main_loop()
        main_loop.run()