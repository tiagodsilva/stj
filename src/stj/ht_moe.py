import jax
import jax.numpy as jnp
from flax import nnx

from stj.nets import MLP

default_init = nnx.initializers.lecun_normal()


class HeterogeneousMoE(nnx.Module):
    def __init__(
        self,
        din: int,
        dmid: int,
        douts: list[int],
        nlayers: int = 2,
        *,
        rngs: nnx.Rngs,
    ):
        self.din = din
        self.dmid = dmid
        self.douts = jnp.array(douts)
        self.num_experts = len(self.douts)

        self.encoder = MLP(self.din, self.dmid, nlayers=nlayers, rngs=rngs)

        dout_max = max(self.douts)
        self.mask = (
            self.douts[:, None] > jnp.arange(dout_max)[None, :]
        )  # (self.num_experts, dout_max)
        self.decoder_weights = nnx.Param(
            default_init(rngs(), (self.num_experts, self.dmid, dout_max))
            * self.mask[:, None, :]
        )

    def __call__(self, x: jax.Array, i: jax.Array):
        y = self.encoder(x)
        w = self.decoder_weights.get_value()

        perm = jnp.argsort(i)
        inv_perm = jnp.argsort(perm)
        group_sizes = jnp.bincount(i, length=self.num_experts)
        y = jax.lax.ragged_dot(y, w * self.mask[:, None, :], group_sizes)
        y = y[inv_perm]
        y = nnx.softmax(jnp.where(self.mask[i], y, -jnp.inf))
        return y


if __name__ == "__main__":
    D = 32
    H = 128
    A = [2, 4, 6]
    B = 32

    seed = 42

    rngs = nnx.Rngs(seed)

    x = jax.random.normal(rngs(), (B, D))
    i = jax.random.categorical(rngs(), jnp.ones((len(A),)), shape=(B,))
    moe = HeterogeneousMoE(D, H, A, rngs=rngs)

    print(moe(x, i))
