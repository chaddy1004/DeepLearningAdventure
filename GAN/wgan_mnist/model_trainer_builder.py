from network.discriminator import Discriminator
from network.discriminator_combined import DiscriminatorCombined
from network.gan_combined import GanCombined
from network.generator import Generator
from trainers.combined_trainer import CombinedTrainer


def build_model_and_trainer(config):
    generator = Generator(config).define_model(model_name="generator")
    discriminator = Discriminator(config).define_model(model_name="discriminator")

    combined_discriminator, parallel_combined_discriminator = DiscriminatorCombined(config).build_model(
        discriminator=discriminator, model_name="discriminator_combined")

    combined, parallel_combined = GanCombined(config).build_model(generator=generator, discriminator=discriminator,
                                                          model_name="gan_combined")

    trainer = CombinedTrainer(generator=generator,
                              discriminator=combined_discriminator,
                              parallel_discriminator=parallel_combined_discriminator,
                              combined=combined,
                              parallel_combined=parallel_combined,
                              config=config)

    return combined, trainer
