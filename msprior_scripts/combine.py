import cached_conv as cc
import nn_tilde
import torch
from absl import flags


class CombinedModule(nn_tilde.Module):

    def __init__(self, prior, rave) -> None:
        super().__init__()
        self.prior_model = torch.jit.load(prior).eval()
        self.rave_model = torch.jit.load(rave).eval()

        self.register_attribute("temperature",
                                self.prior_model.get_temperature())
        self.register_attribute("listen", self.prior_model.get_listen())
        self.register_attribute("learn_context",
                                self.prior_model.get_learn_context())
        self.register_attribute("reset", self.prior_model.get_reset())

        self.register_method(
            "encode",
            *self.rave_model.encode_params,
            self.rave_model.encode_input_labels,
            self.rave_model.encode_output_labels,
        )
        self.register_method(
            "decode",
            *self.rave_model.decode_params,
            self.rave_model.decode_input_labels,
            self.rave_model.decode_output_labels,
        )
        self.register_method(
            "forward",
            *self.rave_model.forward_params,
            self.rave_model.forward_input_labels,
            self.rave_model.forward_output_labels,
        )
        self.register_method(
            "prior",
            *self.prior_model.forward_params,
            self.prior_model.forward_input_labels,
            self.prior_model.forward_output_labels,
        )

    @torch.jit.export
    def encode(self, x):
        return self.rave_model.encode(x)

    @torch.jit.export
    def decode(self, x):
        return self.rave_model.decode(x)

    @torch.jit.export
    def forward(self, x):
        return self.rave_model.forward(x)

    @torch.jit.export
    def prior(self, x):
        return self.prior_model.forward(x)

    @torch.jit.export
    def get_temperature(self) -> float:
        return self.prior_model.get_temperature()

    @torch.jit.export
    def set_temperature(self, temperature: float) -> int:
        return self.prior_model.set_temperature(temperature)

    @torch.jit.export
    def get_listen(self) -> bool:
        return self.prior_model.get_listen()

    @torch.jit.export
    def set_listen(self, listen: bool) -> int:
        return self.prior_model.set_listen(listen)

    @torch.jit.export
    def get_learn_context(self) -> bool:
        return self.prior_model.get_learn_context()

    @torch.jit.export
    def set_learn_context(self, learn_context: bool) -> int:
        return self.prior_model.set_learn_context(learn_context)

    @torch.jit.export
    def get_reset(self) -> bool:
        return self.prior_model.get_reset()

    @torch.jit.export
    def set_reset(self, reset: bool) -> int:
        return self.prior_model.set_reset(reset)


def main(argv):
    cc.MAX_BATCH_SIZE = FLAGS.batch_size
    CombinedModule(FLAGS.prior, FLAGS.rave).export_to_ts(FLAGS.name + ".ts")


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'prior',
    default=None,
    required=True,
    help='Prior model',
)
flags.DEFINE_string(
    'rave',
    default=None,
    help='Streaming rave model',
)
flags.DEFINE_integer(
    'batch_size',
    default=1,
    help='Maximum batch size',
)
flags.DEFINE_string(
    'name',
    default=None,
    required=True,
    help="Combined model name",
)
