import time

import flax.nnx as nnx
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

    def _forward_single(self, x: jax.Array):
        logits = self.clf(x)
        values, indices = jax.lax.top_k(logits, self.K)
        values = nnx.softmax(values, axis=-1)
        y = jax.vmap(self.moe_layer, in_axes=(0, 0), out_axes=0)(
            jnp.broadcast_to(x[None], (self.K, *x.shape)), indices
        )
        return jnp.sum(values[:, None] * y, axis=0)

    def __call__(self, x: jax.Array):
        if x.ndim > 1:
            return jax.vmap(self._forward_single)(x)
        return self._forward_single(x)


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

        y_broad = jnp.broadcast_to(
            y[:, None, :], (y.shape[0], self.K, self.dmid)
        )

        def forward(y: jax.Array, w: jax.Array):
            y = jnp.einsum("bki,bkij->bkj", y, w[indices])
            y = nnx.leaky_relu(y)
            return y, None

        y, _ = jax.lax.scan(
            forward,
            init=y_broad,
            xs=self.moe_w.get_value(),
            length=self.expert_layers,
        )

        return jnp.sum(y * values[..., None], axis=1)


class GatedLowRankMoE(nnx.Module):
    def __init__(
        self,
        din: int,
        dmid: int,
        num_experts: int = 8,
        K: int = 1,
        rank: int = 32,
        expert_layers: int = 4,
        *,
        rngs: nnx.Rngs,
        initializer=None,
    ):
        initializer = initializer or nnx.initializers.lecun_normal()

        self.din = din
        self.dmid = dmid
        self.num_experts = num_experts
        self.K = K
        self.rank = rank
        self.expert_layers = expert_layers

        self.linear_in = nnx.Linear(self.din, self.dmid, rngs=rngs)
        self.clf = nnx.Linear(self.din, self.num_experts, rngs=rngs)

        self.moe_u = nnx.Param(
            initializer(
                rngs(),
                (self.expert_layers, self.num_experts, self.dmid, self.rank),
            )
        )
        self.moe_v = nnx.Param(
            initializer(
                rngs(),
                (self.expert_layers, self.num_experts, self.rank, self.dmid),
            )
        )

    def __call__(self, x: jax.Array):
        unbatched = x.ndim == 1
        if unbatched:
            x = x[None]

        logits = self.clf(x)
        values, indices = jax.lax.top_k(logits, self.K)
        values = nnx.softmax(values, axis=-1)

        y = nnx.leaky_relu(self.linear_in(x))

        def forward(y: jax.Array, uv: tuple[jax.Array, jax.Array]):
            u, v = uv
            h = nnx.leaky_relu(jnp.einsum("bd,bkdr->bkr", y, u[indices]))
            dy = jnp.einsum("bkr,bkrd,bk->bd", h, v[indices], values)
            return y + dy, None

        y, _ = jax.lax.scan(
            forward,
            init=y,
            xs=(self.moe_u.get_value(), self.moe_v.get_value()),
            length=self.expert_layers,
        )

        return y[0] if unbatched else y


def count_params(model: nnx.Module) -> int:
    params = nnx.state(model, nnx.Param)
    return sum(p.size for p in jax.tree.leaves(params))


def count_flops(model: nnx.Module, x: jax.Array) -> int:
    og_scan = jax.lax.scan
    try:
        jax.lax.scan = lambda f, init, xs=None, length=None, reverse=False: (
            og_scan(
                f, init, xs=xs, length=length, reverse=reverse, unroll=True
            )
        )
        ca = jax.jit(model).lower(x).cost_analysis()
        return ca["flops"]
    finally:
        jax.lax.scan = og_scan


def build_models(din: int, dmid: int, seed: int):
    return [
        (
            "MLP (L=4)",
            4,
            "-",
            MLP(din, dmid, nlayers=4, rngs=nnx.Rngs(seed)),
        ),
        (
            "MLP (L=8)",
            8,
            "-",
            MLP(din, dmid, nlayers=8, rngs=nnx.Rngs(seed)),
        ),
        (
            "RaggedMoE (E=8, K=1, L=4)",
            4,
            8,
            GatedMoELayerRagged(
                din,
                dmid,
                num_experts=8,
                K=1,
                nlayers=2,
                expert_layers=2,
                rngs=nnx.Rngs(seed),
            ),
        ),
        (
            "NonRaggedMoE (E=8, K=1, L=4)",
            4,
            8,
            GatedMoELayerNonRagged(
                din,
                dmid,
                num_experts=8,
                K=1,
                nlayers=2,
                expert_layers=2,
                rngs=nnx.Rngs(seed),
            ),
        ),
        (
            "RaggedMoE (E=16, K=1, L=4)",
            4,
            16,
            GatedMoELayerRagged(
                din,
                dmid,
                num_experts=16,
                K=1,
                nlayers=2,
                expert_layers=2,
                rngs=nnx.Rngs(seed),
            ),
        ),
        (
            "NonRaggedMoE (E=16, K=1, L=4)",
            4,
            16,
            GatedMoELayerNonRagged(
                din,
                dmid,
                num_experts=16,
                K=1,
                nlayers=2,
                expert_layers=2,
                rngs=nnx.Rngs(seed),
            ),
        ),
        (
            "RaggedMoE (E=16, K=1, L=3)",
            3,
            16,
            GatedMoELayerRagged(
                din,
                dmid,
                num_experts=16,
                K=1,
                nlayers=1,
                expert_layers=2,
                rngs=nnx.Rngs(seed),
            ),
        ),
        (
            "NonRaggedMoE (E=16, K=1, L=3)",
            3,
            16,
            GatedMoELayerNonRagged(
                din,
                dmid,
                num_experts=16,
                K=1,
                nlayers=1,
                expert_layers=2,
                rngs=nnx.Rngs(seed),
            ),
        ),
        (
            "RaggedMoE (E=8, K=1, L=3)",
            3,
            8,
            GatedMoELayerRagged(
                din,
                dmid,
                num_experts=8,
                K=1,
                nlayers=1,
                expert_layers=2,
                rngs=nnx.Rngs(seed),
            ),
        ),
        (
            "NonRaggedMoE (E=8, K=1, L=3)",
            3,
            8,
            GatedMoELayerNonRagged(
                din,
                dmid,
                num_experts=8,
                K=1,
                nlayers=1,
                expert_layers=2,
                rngs=nnx.Rngs(seed),
            ),
        ),
        (
            "LowRankMoE (E=8, K=1, r=32, L=4)",
            4,
            8,
            GatedLowRankMoE(
                din,
                dmid,
                num_experts=8,
                K=1,
                rank=32,
                expert_layers=4,
                rngs=nnx.Rngs(seed),
            ),
        ),
        (
            "LowRankMoE (E=8, K=2, r=32, L=4)",
            4,
            8,
            GatedLowRankMoE(
                din,
                dmid,
                num_experts=8,
                K=2,
                rank=32,
                expert_layers=4,
                rngs=nnx.Rngs(seed),
            ),
        ),
        (
            "LowRankMoE (E=16, K=1, r=32, L=4)",
            4,
            16,
            GatedLowRankMoE(
                din,
                dmid,
                num_experts=16,
                K=1,
                rank=32,
                expert_layers=4,
                rngs=nnx.Rngs(seed),
            ),
        ),
        (
            "LowRankMoE (E=16, K=2, r=32, L=4)",
            4,
            16,
            GatedLowRankMoE(
                din,
                dmid,
                num_experts=16,
                K=2,
                rank=32,
                expert_layers=4,
                rngs=nnx.Rngs(seed),
            ),
        ),
    ]


def benchmark_runtime(f: callable, xs: jax.Array, warmup: int = 8) -> float:
    for _ in range(warmup):
        f(xs[-1]).block_until_ready()
    t = time.monotonic_ns()
    _, ys = jax.lax.scan(lambda y, x: (y, f(x)), init=xs, xs=xs)
    ys.block_until_ready()
    return (time.monotonic_ns() - t) / 1e6


def main():
    N = 32
    D = 128
    B = 64
    seed = 42

    key = jax.random.key(seed)
    x = jax.random.normal(key, (B, N))

    models = build_models(N, D, seed)

    ragged = GatedMoELayerRagged(
        N,
        D,
        num_experts=8,
        K=1,
        nlayers=2,
        expert_layers=2,
        rngs=nnx.Rngs(seed),
    )
    non_ragged = GatedMoELayerNonRagged(
        N,
        D,
        num_experts=8,
        K=1,
        nlayers=2,
        expert_layers=2,
        rngs=nnx.Rngs(seed),
    )
    print(
        "Verification:",
        "ragged shape:",
        ragged(x).shape,
        "| non-ragged shape:",
        non_ragged(x).shape,
        "| match:",
        bool(jnp.isclose(ragged(x), non_ragged(x), atol=1e-5).all()),
    )

    repeat = 128
    xs = jax.random.normal(key, (repeat, B, N))
    print(
        f"\n{'Model':<38} | {'Depth':>5} | {'E':>3} | {'Params':>10} | {'Runtime (32 calls, ms)':>22} | {'Per call (ms)':>13} | {'Flops':>10}"
    )
    print("-" * 125)
    for name, depth, e_val, model in models:
        params = count_params(model)
        flops = count_flops(model, xs[-1])
        total_ms = benchmark_runtime(jax.jit(model), xs)
        per_call_ms = total_ms / repeat
        print(
            f"{name:<38} | {depth:>5} | {e_val:>3} | {params:>10,d} | {total_ms:>22.2f} | {per_call_ms:>13.3f}  | {flops:>10.2e}"
        )


if __name__ == "__main__":
    main()
