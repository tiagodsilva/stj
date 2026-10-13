import flax.nnx as nnx
import flax.struct as struct
import jax
import jax.numpy as jnp

from stj.nets import MLP


class MoELayer(nnx.Module):
    def __init__(
        self,
        din: int,
        dmid: int,
        num_experts: int,
        *,
        rngs: nnx.Rngs,
        expert_kwargs: None | dict = None,
    ):
        expert_kwargs = expert_kwargs or {"nlayers": 2}

        self.din = din
        self.dmid = dmid

        self.num_experts = num_experts

        def build_expert(rngs):
            return MLP(self.din, self.dmid, **expert_kwargs, rngs=rngs)

        @nnx.split_rngs(splits=num_experts)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_layer(rngs):
            return build_expert(rngs=rngs)

        self.expert_graphdef, _ = nnx.split(build_expert(rngs=rngs))

        self.experts = create_layer(rngs)

    # This is applies to an unbatched state; vmapping is necessary
    def __call__(self, x: jax.Array, exp: jax.Array):
        _, expert_states = nnx.split(self.experts)
        gathered = jax.tree.map(lambda leaf: leaf[exp], expert_states)
        expert = nnx.merge(self.expert_graphdef, gathered)
        return expert(x)


class GatedMoETopK(nnx.Module):
    def __init__(
        self,
        din: int,
        dmid: int,
        num_experts: int,
        K: int,
        *,
        rngs: nnx.Rngs,
        expert_kwargs=None,
    ):
        self.din = din
        self.dmid = dmid
        self.num_experts = num_experts
        self.K = K

        self.moe_layer = MoELayer(
            din, dmid, num_experts, rngs=rngs, expert_kwargs=expert_kwargs
        )

        # Classifier for computing the logits
        self.clf = nnx.Linear(self.din, self.num_experts, rngs=rngs)

    def __call__(self, x: jax.Array):
        logits = self.clf(x)
        (values, indices) = jax.lax.top_k(logits, self.K)
        values = nnx.softmax(values, axis=-1)
        y = jax.vmap(self.moe_layer, in_axes=(0, 0), out_axes=0)(
            jnp.broadcast_to(x[None], (self.K, *x.shape)), indices
        )
        return jnp.sum(values[:, None] * y, axis=0)


if __name__ == "__main__":
    N = 32
    D = 128
    E = 4
    B = 64
    seed = 42

    key = jax.random.key(seed)
    rngs = nnx.Rngs(key)

    x = jax.random.normal(key, shape=(N,))
    exp = jnp.array(E // 2, dtype=jnp.int32)

    moe = MoELayer(N, D, E, rngs=rngs)
    print(moe(x, exp).shape)

    # Check out vmapped operations
    x = jnp.broadcast_to(jax.random.normal(key, shape=(1, N)), (B, N))
    exp = jax.random.categorical(key, jnp.ones((E,)), shape=(B,))
    print(jax.vmap(moe, in_axes=(0, 0), out_axes=0)(x, exp), exp)

    # Check out the Gated Layer
    x = jax.random.normal(key, (N,))
    K = E // 2
    moe_gated = GatedMoETopK(N, D, E, K, rngs=rngs)
    print(moe_gated(x).shape)
