import flax.nnx as nnx
import jax
import jax.numpy as jnp

_act = nnx.leaky_relu
_int = nnx.initializers.lecun_normal()


class Attention(nnx.Module):
    def __init__(self, din: int, dmid: int, *, rngs: nnx.Rngs):
        self.din = din
        self.dmid = dmid

        self.proj_q = nnx.Linear(self.din, self.dmid, rngs=rngs)
        self.proj_k = nnx.Linear(self.din, self.dmid, rngs=rngs)
        self.proj_v = nnx.Linear(self.din, self.dmid, rngs=rngs)

        self.sqrt_d = jnp.sqrt(self.dmid)

    @staticmethod
    def _get_mask(x: jax.Array, mask: jax.Array):
        if mask is None:
            return jnp.ones((x.shape[0],), dtype=jnp.bool)
        return mask

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        mq: jax.Array = None,  # Binary mask
        mk: jax.Array = None,  # Binary mask
    ):
        (nq, _) = q.shape
        (nk, _) = k.shape

        if mq is None:
            mq = jnp.ones((nq,), dtype=jnp.bool)
        if mk is None:
            mk = jnp.ones((nk,), dtype=jnp.bool)

        attn_mask = jnp.where(mk[None, :], 0, -jnp.inf)
        # x -> MAB(x, x)
        # x in (B, dim): softmax ( (x W_{Q})^{T} (x W_{K}) ) (x W_V)
        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v)
        y = (
            nnx.softmax((q @ k.T) / self.sqrt_d + attn_mask, axis=-1) @ v
        )  # (S, dim)
        y = jnp.where(mq[:, None], y, 0)
        return y


class MultiheadSelfAttention(nnx.Module):
    def __init__(
        self, din: int, dmid: int, dout: int, h: int, *, rngs: nnx.Rngs
    ):
        self.din = din
        self.dmid = dmid
        self.h = h
        self.dout = dout

        @nnx.split_rngs(splits=h)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_attn(rngs: nnx.Rngs):
            return Attention(self.din, self.dmid // self.h, rngs=rngs)

        self.attn_heads = create_attn(rngs)
        self.proj = nnx.Linear(self.dmid, self.dout, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        y: jax.Array,
        mq: jax.Array = None,
        mk: jax.Array = None,
    ):
        def forward_fn(layer: Attention, x: jax.Array, y: jax.Array):
            return layer(x, y, y, mq, mk)

        attn = jax.vmap(forward_fn, in_axes=(0, None, None), out_axes=1)(
            self.attn_heads, x, y
        )  # (S, h, dmid)
        attn = jnp.reshape(attn, (-1, self.dmid))
        y = self.proj(attn)
        return y  # (S, dout)


class LayerNorm(nnx.Module):
    def __init__(self, d: int, axis: int = -1, eps: float = 1e-6):
        self.d = d
        self.axis = axis
        self.eps = eps
        self.gamma = nnx.Param(jnp.ones((d,)))
        self.beta = nnx.Param(jnp.ones((d,)))

    def __call__(self, x: jax.Array):
        var = jnp.var(x, keepdims=True, axis=self.axis)
        std = jnp.sqrt(var + self.eps)
        mu = jnp.mean(x, keepdims=True, axis=1)
        return self.gamma[None] * (x - mu) / std + self.beta[None]


def forward(layers: nnx.Module, x: jax.Array, nlayers: int, **kwargs):
    def forward_fn(y: jax.Array, layer: nnx.Module):
        return _act(layer(y, **kwargs)), None

    y, _ = jax.lax.scan(forward_fn, length=nlayers, init=x, xs=layers)
    return y


class MLP(nnx.Module):
    def __init__(self, din: int, dmid: int, nlayers: int, *, rngs: nnx.Rngs):
        self.din = din
        self.dmid = dmid
        self.nlayers = nlayers

        @nnx.split_rngs(splits=self.nlayers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_layer(rngs: nnx.Rngs):
            return nnx.Linear(self.dmid, self.dmid, rngs=rngs)

        self.linear_in = nnx.Linear(self.din, self.dmid, rngs=rngs)
        self.layers = create_layer(rngs)

    def __call__(self, x: jax.Array):
        y = self.linear_in(x)
        y = _act(y)
        y = forward(self.layers, y, self.nlayers)
        return y


class SetTransformerLayer(nnx.Module):
    def __init__(
        self,
        dmid: int,
        dout: int,
        h: int,
        nlayers_ff: int = 2,
        n_inducing: int | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.dmid = dmid
        self.dout = dout
        self.h = h
        self.n_inducing = n_inducing

        if self.n_inducing:
            self.ips = nnx.Param(_int(rngs(), (self.n_inducing, self.dmid)))

        self.mha = MultiheadSelfAttention(
            self.dmid, self.dmid, self.dout, h=self.h, rngs=rngs
        )
        self.ln_in = LayerNorm(self.dmid)
        self.ln_out = LayerNorm(self.dmid)

        self.ff = MLP(self.dmid, self.dmid, nlayers=nlayers_ff, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array = None):
        mq = Attention._get_mask(x, mask)
        if self.n_inducing:
            mk = jnp.ones((self.n_inducing,), dtype=jnp.bool)
            h = self.ln_in(x + self.mha(x, self.ips, mq, mk))
        else:
            mk = mq
            h = self.ln_in(x + self.mha(x, x, mq, mk))
        y = self.ln_out(h + self.ff(h))
        return y


class SetTransformerEncoder(nnx.Module):
    def __init__(
        self,
        din: int,
        dmid: int,
        h: int,
        n_inducing: int | None = None,
        nlayers: int = 2,
        # If max_items is unknown (or infinite), use a linear embedding
        max_items: int | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.din = din
        self.dmid = dmid
        self.h = h
        self.nlayers = nlayers
        self.max_items = max_items

        @nnx.split_rngs(splits=self.nlayers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_layer(rngs: nnx.Rngs):
            return SetTransformerLayer(
                self.dmid, self.dmid, self.h, n_inducing=n_inducing, rngs=rngs
            )

        if self.max_items:
            self.linear_in = nnx.Embed(self.max_items, self.dmid, rngs=rngs)
        else:
            self.linear_in = nnx.Linear(self.din, self.dmid, rngs=rngs)
        self.layers = create_layer(rngs)

    def __call__(self, x: jax.Array, mask: jax.Array = None):
        mask = Attention._get_mask(x, mask)
        y = jnp.where(mask[:, None], self.linear_in(x), 0)
        y = forward(self.layers, y, self.nlayers, mask=mask)
        return y


class SetTransformerDecoder(nnx.Module):
    def __init__(
        self,
        dmid: int,
        dout: int,
        nlayers_ff: int = 2,
        k: int = 1,
        h: int = 4,
        *,
        rngs: nnx.Rngs,
    ):
        self.dmid = dmid
        self.dout = dout
        self.k = k

        self.queries = nnx.Param(_int(rngs(), (self.k, self.dmid)))
        self.mha = MultiheadSelfAttention(dmid, dmid, dmid, h=h, rngs=rngs)
        self.layer = SetTransformerLayer(dmid, dmid, h=h, rngs=rngs)

        self.ff = MLP(dmid, dout, nlayers=nlayers_ff, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array = None):
        mq = jnp.ones((self.k,), dtype=jnp.bool)
        mk = Attention._get_mask(x, mask)

        y = self.mha(self.queries, x, mq=mq, mk=mk)
        y = self.layer(y)
        y = self.ff(y)
        return y


class SetTransformer(nnx.Module):
    def __init__(
        self,
        din: int,
        dmid: int,
        dout: int,
        nlayers: int = 2,
        h: int = 4,
        nlayers_ff: int = 2,
        n_inducing: int | None = None,
        max_items: int | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.din = din
        self.dmid = dmid
        self.dout = dout
        self.nlayers = nlayers
        self.h = h
        self.nlayers_ff = nlayers_ff
        self.n_inducing = n_inducing

        self.encoder = SetTransformerEncoder(
            self.din,
            self.dmid,
            self.h,
            self.n_inducing,
            self.nlayers_ff,
            max_items=max_items,
            rngs=rngs,
        )
        self.decoder = SetTransformerDecoder(
            self.dmid, self.dout, self.nlayers_ff, h=self.h, rngs=rngs
        )

    def __call__(self, x: jax.Array, mask: jax.Array = None):
        nx = x.shape[0]
        if mask is None:
            mask = jnp.ones((nx,), dtype=jnp.bool)

        y = self.encoder(x, mask)
        y = self.decoder(y, mask)
        return y


if __name__ == "__main__":
    din = 8
    dmid = 128
    dout = 6
    S = 64
    h = 4

    key = jax.random.key(42)
    rngs = nnx.Rngs(key)

    self_attn = Attention(din, dmid, rngs=rngs)
    mha = MultiheadSelfAttention(din, dmid, dout, h=h, rngs=rngs)

    x = jax.random.normal(key, (S, din))
    print(self_attn(x, x, x).shape)
    print(mha(x, x).shape)

    encoder = SetTransformerEncoder(din, dmid, h=h, rngs=rngs)
    decoder = SetTransformerDecoder(dmid, dout, rngs=rngs)

    y = encoder(x)
    print(y.shape)
    y = decoder(y)
    print(y.shape)

    encoder_inducing = SetTransformerEncoder(
        din, dmid, h=h, n_inducing=8, rngs=rngs
    )

    y = encoder_inducing(x)
    print(y.shape)
    y = decoder(y)
    print(y.shape)

    set_trans = SetTransformer(din, dmid, dout, rngs=rngs)
    print(set_trans(x))

    # Use categorical variables
    max_items = 9
    x_cat = jax.random.categorical(
        key, logits=jnp.ones((max_items,)), shape=(32, S)
    )
    set_trans_cat = SetTransformer(
        din, dmid, dout, n_inducing=8, max_items=max_items, rngs=rngs
    )
    print(jax.vmap(set_trans_cat, in_axes=(0,), out_axes=0)(x_cat).shape)

    # Check out masked transformer
    mask = jnp.ones((S,), dtype=jnp.bool)
    mask = mask.at[S // 2 :].set(False)
    print(set_trans(x, mask))
