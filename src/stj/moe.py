import time

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
        self.clf = nnx.Linear(self.dmid, self.num_experts, rngs=rngs)

    def __call__(self, x: jax.Array):
        logits = self.clf(x)
        (values, indices) = jax.lax.top_k(logits, self.K)
        values = nnx.softmax(values, axis=-1)
        y = jax.vmap(self.moe_layer, in_axes=(0, 0), out_axes=0)(
            jnp.broadcast_to(x[None], (self.K, *x.shape)), indices
        )
        return jnp.sum(values[:, None] * y, axis=0)


class GatedMoELayerRagged(nnx.Module):
    def __init__(
        self,
        din: int,
        dmid: int,
        num_experts: int,
        K: int,
        nlayers: int = 2,
        expert_layers: int = 2,
        *,
        rngs: nnx.Rngs,
        initializer=None,
    ):
        initializer = initializer or nnx.initializers.lecun_normal()

        self.din = din
        self.dmid = dmid
        self.num_experts = num_experts
        self.K = K

        self.nlayers = nlayers
        self.expert_layers = expert_layers

        self.encoder = MLP(
            self.din, self.dmid, nlayers=self.nlayers, rngs=rngs
        )

        self.moe_w = nnx.Param(
            initializer(
                rngs(),
                (self.expert_layers, self.num_experts, self.dmid, self.dmid),
            )
        )

        self.clf = nnx.Linear(self.dmid, self.num_experts, rngs=rngs)

    def __call__(self, x: jax.Array):
        y = self.encoder(x)
        logits = self.clf(y)
        (values, indices) = jax.lax.top_k(logits, self.K)
        values = nnx.softmax(values, axis=-1)

        # We evaluate K experts per sample
        expert_idx = indices.reshape(-1)
        y_broad = jnp.broadcast_to(
            y[:, None, :], (y.shape[0], self.K, self.dmid)
        ).reshape(-1, self.dmid)

        perm = jnp.argsort(expert_idx)
        inv_perm = jnp.argsort(perm)

        group_sizes = jnp.bincount(expert_idx, length=self.num_experts)

        def forward(y: jax.Array, w: jax.Array):
            y = jax.lax.ragged_dot(y, w, group_sizes)
            y = nnx.leaky_relu(y)
            return y, None

        y, _ = jax.lax.scan(
            forward,
            init=y_broad[perm],
            xs=self.moe_w.get_value(),
            length=self.expert_layers,
        )

        y = y[inv_perm].reshape(-1, self.K, self.dmid)
        y = jnp.sum(y * values[..., None], axis=1)
        return y


class GatedMoELayerNonRagged(GatedMoELayerRagged):
    def __call__(self, x: jax.Array):
        y = self.encoder(x)
        logits = self.clf(y)
        (values, indices) = jax.lax.top_k(logits, self.K)
        values = nnx.softmax(values, axis=-1)

        def forward(y: jax.Array, w: jax.Array):
            y = jnp.einsum("bi,bij->bj", y, w[indices])
            y = nnx.leaky_relu(y)
            return y, None

        y, _ = jax.lax.scan(
            forward,
            init=y,
            xs=self.moe_w.get_value(),
            nlayers=self.expert_layers,
        )

        return jnp.sum(y * values[..., None], axis=1)


if __name__ == "__main__":
    N = 32
    D = 512
    E = 8
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
    x = jax.random.normal(key, (B, N))
    K = E // 2
    moe_gated = GatedMoETopK(N, D, E, K, rngs=rngs)
    print(moe_gated(x).shape)

    # Ragged versions
    moe_gated_ragged = GatedMoELayerRagged(N, D, E, K, rngs=nnx.Rngs(seed))
    moe_gated_non_ragged = GatedMoELayerRagged(N, D, E, K, rngs=nnx.Rngs(seed))

    print(
        moe_gated_ragged(x).shape,
        moe_gated_non_ragged(x).shape,
        jnp.isclose(moe_gated_ragged(x), moe_gated_non_ragged(x)).all(),
    )

    def runtime(f: callable, repeat: int = 32, warmup: int = 8):
        for _ in range(warmup):
            f(x)  # run a few times to compile
        t = time.monotonic_ns()
        for _ in range(repeat):
            y: jax.Array = f(x)
            y.block_until_ready()

        return (time.monotonic_ns() - t) / 1e6

    print(runtime(jax.jit(moe_gated_ragged)))
    print(runtime(jax.jit(moe_gated_non_ragged)))
