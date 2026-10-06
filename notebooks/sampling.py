import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import optax

    import flax.nnx as nnx
    import flax.struct as struct
    import tqdm

    from functools import partial

    from stj.nets import SetTransformer, MLP

    return MLP, SetTransformer, jax, jnp, nnx, optax, partial, struct, tqdm


@app.cell
def _(jax, jnp, struct):
    NIL = -1

    @struct.dataclass
    class EnvState:
        num_items: int = struct.field(pytree_node=False)

        state: jax.Array
        items: jax.Array

        fmask: jax.Array
        bmask: jax.Array

        fstopped: jax.Array
        bstopped: jax.Array

        curr_idx: jax.Array

        batch_ids: jax.Array
        bs: int = struct.field(pytree_node=False)

        @staticmethod
        def create(bs: int, num_items: int):
            state = jnp.zeros((bs, num_items))
            items = jnp.ones((bs, num_items)) * NIL

            fmask = jnp.ones((bs, num_items + 1))
            bmask = jnp.zeros((bs, num_items + 1))

            return EnvState(
                num_items=num_items,
                state=state,
                items=items,
                fmask=fmask,
                bmask=bmask,
                fstopped=jnp.zeros((bs,), dtype=jnp.bool),
                bstopped=jnp.ones((bs,), dtype=jnp.bool),
                curr_idx=jnp.zeros((bs,), dtype=jnp.int32),
                batch_ids=jnp.arange(bs),
                bs=bs,
            )

    return EnvState, NIL


@app.cell
def _(EnvState, MLP, NIL, SetTransformer, jax, jnp, nnx):
    class Policy(nnx.Module):
        def __init__(
            self,
            num_items: int,
            dmid: int,
            nlayers: int = 2,
            n_inducing: int | None = None,
            *,
            rngs: nnx.Rngs,
        ):
            self.num_items = num_items
            self.dmid = dmid
            self.logz = nnx.Param(jnp.array(0.0))

            self.model = SetTransformer(
                1,
                dmid,
                self.num_items + 1,
                nlayers=nlayers,
                n_inducing=n_inducing,
                # A token is the start-of-sentence token
                max_items=self.num_items + 1,
                rngs=rngs,
            )

        def __call__(self, x: EnvState):
            i = x.items  # (B, s)
            i = jnp.hstack([jnp.ones((x.bs, 1)) * self.num_items, i])
            i = jnp.astype(i, jnp.int32)
            m = i != NIL

            l = jax.vmap(self.model, in_axes=(0, 0), out_axes=0)(i, m)
            l = l.squeeze(axis=1)
            l = jnp.where(x.fmask == 1, l, -jnp.inf)
            l = nnx.log_softmax(l, axis=-1)
            return l

    class PolicyMLP(nnx.Module):
        def __init__(
            self,
            num_items: int,
            dmid: int,
            nlayers: int = 2,
            *,
            rngs: nnx.Rngs,
        ):
            self.logz = nnx.Param(jnp.array(0.0))
            self.num_items = num_items
            self.model = MLP(num_items, dmid, nlayers=nlayers, rngs=rngs)
            self.linear = nnx.Linear(dmid, num_items + 1, rngs=rngs)

        def __call__(self, x: EnvState):
            l = self.linear(nnx.leaky_relu(self.model(x.state)))
            return nnx.log_softmax(
                jnp.where(x.fmask == 1, l, -jnp.inf), axis=1
            )

    return Policy, PolicyMLP


@app.cell
def _(EnvState, jax, jnp):
    def umask(
        fmask: jax.Array,
        bmask: jax.Array,
        ohe: jax.Array,
        is_stop_action: jax.Array,
        num_items: int,
    ):
        fmask_bool = fmask.astype(jnp.bool)
        bmask_bool = bmask.astype(jnp.bool)

        isa = is_stop_action[:, None]
        ial = (ohe.sum(axis=1) >= num_items)[:, None]  # is above limit
        ii = ohe == 1  # is included
        ni = num_items

        fmask = fmask_bool.at[:, :ni].set(~isa & ~ii & ~ial)

        bmask = bmask_bool.at[:, :num_items].set(ii & ~isa)
        bmask = bmask.at[:, num_items].set(is_stop_action)

        return fmask.astype(jnp.float32), bmask.astype(jnp.float32)

    def fapply(state: EnvState, actions: jax.Array):
        isa = actions == state.num_items
        bds = state.batch_ids
        ci = state.curr_idx
        ni = state.num_items

        i = state.items

        cs = state.state[bds, actions]
        ns = state.state.at[bds, actions].set(jnp.where(isa, cs, 1))

        # Update masks
        fm, bm = umask(state.fmask, state.bmask, ns, isa, ni)

        # Update flags
        fstopped = jnp.where(isa, True, state.fstopped)
        bstopped = jnp.zeros_like(state.bstopped)

        i = i.at[bds, ci].set(jnp.where(isa, i[bds, ci], actions))
        ci = ci + jnp.astype(~isa, ci.dtype)

        # Return the updated state
        return state.replace(
            state=ns, items=i, fmask=fm, bmask=bm, curr_idx=ci
        )

    return (fapply,)


@app.cell(hide_code=True)
def _(EnvState, Policy, fapply, jax, jnp, nnx):
    def fstep(
        carry: tuple[EnvState, jax.Array], _, pol: Policy, eps: float = 5e-2
    ):
        state, key = carry
        bids = state.batch_ids

        # Compute model probabilities
        flogits = pol(state)

        # Compute the exploratory policy
        model_probs = jnp.exp(flogits)
        uniform_probs = nnx.softmax(
            jnp.where(state.fmask == 1, 1, -jnp.inf), axis=-1
        )

        # Compute logits and sample actions
        slogits = jnp.log(eps * uniform_probs + (1 - eps) * model_probs)

        # Sample actions
        key, subkey = jax.random.split(key, 2)
        a = jax.random.categorical(subkey, slogits)

        fstate = fapply(state, a)

        # Compute transition logits
        blogits = nnx.log_softmax(jnp.where(fstate.bmask == 1, 1, -jnp.inf))

        flogits = jnp.where(state.fstopped, 0, flogits[bids, a])
        blogits = jnp.where(state.fstopped, 0, blogits[bids, a])

        return (fstate, key), (flogits, blogits)

    return (fstep,)


@app.cell
def _(EnvState, jax, jnp):
    # def get_items(num_items: int, r: float = 2, seed: int = 43):
    #     key = jax.random.key(seed)

    #     items = jnp.ones((num_items))
    #     items = items.at[: num_items // 2].set(r)
    #     items = items.at[num_items // 2 :].set(-r)

    #     return jax.random.permutation(key, items)

    def get_items(num_items: int, seed: int = 43):
        key = jax.random.key(seed)
        key, *_ = jax.random.split(key, 3)
        log_u_template = jax.random.normal(key, (num_items,))
        return log_u_template

    def logr(x: EnvState):
        # We use a deterministic policy
        items = get_items(x.num_items)
        return jnp.einsum("bi,i->b", x.state, items)

    return get_items, logr


@app.cell
def _(nnx, optax):
    def create_opt(pol: nnx.Module, logz_lr: float = 1e-1, lr: float = 1e-2):
        params = nnx.state(pol, nnx.Param)

        def label_params(path, _):
            if "logz" in path:
                return "logz"
            return "default"

        param_labels = nnx.map_state(label_params, params)

        tx = optax.multi_transform(
            {"logz": optax.adam(logz_lr), "default": optax.contrib.muon(lr)},
            param_labels,
        )

        return nnx.Optimizer(pol, tx=tx, wrt=nnx.Param)

    return (create_opt,)


@app.cell
def _(
    EnvState,
    Policy,
    fstep,
    get_item_marg,
    get_items,
    jax,
    jnp,
    logr,
    nnx,
    partial,
):
    @partial(nnx.jit, static_argnames=("bs", "num_items"))
    def train_step(
        key: jax.Array,
        pol: Policy,
        opt: nnx.Optimizer,
        bs: int,
        num_items: int,
    ):
        # Create an initial state
        s = EnvState.create(bs, num_items)

        def loss_fn(pol: Policy):
            (x, nk), (flogits, blogits) = jax.lax.scan(
                partial(fstep, pol=pol), init=(s, key), length=num_items
            )
            logrs = logr(x)
            loss = logrs + (blogits - flogits).sum(axis=0) - pol.logz
            loss = jnp.var(loss)
            return loss, nk

        (loss, key), grads = nnx.value_and_grad(
            loss_fn, argnums=0, has_aux=True
        )(pol)
        opt.update(pol, grads)

        return loss, key

    def eval_step(
        key: jax.Array, pol: Policy, bs: int, num_items: int, repeat: int = 32
    ):
        @nnx.jit
        def sample(key: jax.Array, _, s: EnvState):
            (x, key), _ = jax.lax.scan(
                partial(fstep, pol=pol), init=(s, key), length=num_items
            )
            return key, x

        s = EnvState.create(bs, num_items)
        _, x = jax.lax.scan(partial(sample, s=s), init=key, length=repeat)
        state = x.state.reshape(repeat * bs, num_items)
        freqs = state.mean(axis=0)
        probs = get_item_marg(get_items(num_items))

        return jnp.abs(freqs - probs).max()

    return eval_step, train_step


@app.cell
def _():
    num_items = 64
    return (num_items,)


@app.cell(hide_code=True)
def _(jax, jnp, partial):
    def sum_over_fixed_size(log_u: jax.Array):
        N = log_u.shape[0]
        n_cols = N + 1
        M = jnp.full((N, n_cols), fill_value=-jnp.inf, dtype=log_u.dtype)
        M = M.at[:, 0].set(0.0)
        M = M.at[0, 1].set(log_u[0])

        def scan_body(idx, m):
            mprev = m[idx - 1]
            mshift = jnp.concatenate(
                [jnp.array([-jnp.inf], dtype=mprev.dtype), mprev[:-1]]
            )
            new_row = jnp.logaddexp(log_u[idx] + mshift, mprev)
            new_row = new_row.at[0].set(0.0)
            return m.at[idx].set(new_row)

        M = jax.lax.fori_loop(1, N, scan_body, M)
        return M

    def _log_z_from_log_u(log_u: jax.Array):
        M = sum_over_fixed_size(log_u)
        lo, hi = 0, log_u.shape[0] + 1
        contrib = M[-1, lo:hi]
        return jax.nn.logsumexp(contrib, axis=0)

    def get_item_marg(log_u: jax.Array):
        log_u = log_u
        lo = 0
        ms = log_u.shape[0]

        @partial(jax.vmap, in_axes=(0,), out_axes=0)
        def log_z_leave_one_out(log_u_sub: jax.Array):
            M = sum_over_fixed_size(log_u_sub)
            return jax.nn.logsumexp(M[-1, lo:ms], axis=0)

        j = jnp.arange(ms - 1)
        i = jnp.arange(ms)[:, None]
        col_idx = j + (j >= i)
        log_u_matrix = log_u[col_idx]
        log_z = _log_z_from_log_u(log_u)
        log_z_without = log_z_leave_one_out(log_u_matrix)
        return jnp.exp(log_u + log_z_without - log_z)

    return (get_item_marg,)


@app.cell
def _(PolicyMLP, create_opt, eval_step, jax, nnx, num_items, tqdm, train_step):
    def train_mlp(num_items: int, steps: int, bs: int = 32, seed: int = 42):
        key = jax.random.key(seed)
        rngs = nnx.Rngs(key)

        pol = PolicyMLP(num_items, dmid=32, rngs=rngs)
        opt = create_opt(pol, lr=1e-3)

        for _ in (pbar := tqdm.trange(steps)):
            loss, key = train_step(key, pol, opt, bs=bs, num_items=num_items)
            pbar.set_postfix(loss=f"{loss:.2e}")

        return eval_step(key, pol, bs, num_items)

    train_mlp(num_items=num_items, steps=int(1e3))
    return


@app.cell
def _(Policy, create_opt, eval_step, jax, nnx, num_items, tqdm, train_step):
    def train_transformer(
        num_items: int, steps: int, bs: int = 32, seed: int = 42
    ):
        key = jax.random.key(seed)
        rngs = nnx.Rngs(key)

        pol = Policy(num_items, dmid=16, n_inducing=4, nlayers=1, rngs=rngs)
        opt = create_opt(pol, lr=1e-3)

        for _ in (pbar := tqdm.trange(steps)):
            loss, key = train_step(key, pol, opt, bs=bs, num_items=num_items)
            pbar.set_postfix(loss=f"{loss:.2e}")

        return eval_step(key, pol, bs, num_items)

    train_transformer(num_items=num_items, steps=int(1e3))
    return


if __name__ == "__main__":
    app.run()
