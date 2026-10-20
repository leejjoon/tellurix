"""Compiled functions whose arrays are operands, so one compilation serves many profiles.

``jax.jit`` bakes every array a function closes over into the program it
compiles. A model closes over its atmosphere, so two fits of one window with two
profiles -- the same shapes, the same operations -- compile two programs that
differ only in their constants, and the persistent compilation cache, which keys
on the program, never hits. Measured on a Baker et al. 2020 window: 2.2 s to
compile the objective and 1.9 s the opacity precompute, against 0.9 s of fitting.

:func:`hoist_constants` traces the function once and returns it as a function of
those constants plus its own arguments. Compiled, the program is the same for
every profile on that window, so with ``jax_compilation_cache_dir`` set a later
profile loads it instead of compiling. Without the cache nothing is saved and
nothing is lost but one trace (~0.1 s).

Only for a function over precomputed opacity. The line-by-line kernel decides
at compile time that a constant profile is inside its sparse bounds; given the
profile as an operand it compiles the dense fallback too, and a Baker window's
peak memory goes from 0.54 to 4.6 GB.
"""

from __future__ import annotations

import jax


def hoist_constants(fun, *example_args):
    """``(run, constants)`` with ``run(constants, *args) == fun(*args)``.

    ``example_args`` fix the shapes and the structure; the constants are the
    arrays ``fun`` closed over, in the order its jaxpr holds them. ``run`` is a
    plain function, to be wrapped in ``jax.jit`` or a transform of it.
    """

    closed, shape = jax.make_jaxpr(fun, return_shape=True)(*example_args)
    jaxpr = closed.jaxpr
    output = jax.tree_util.tree_structure(shape)

    def run(constants, *args):
        flat = jax.tree_util.tree_leaves(args)
        return jax.tree_util.tree_unflatten(output, jax.core.eval_jaxpr(jaxpr, constants, *flat))

    return run, list(closed.consts)


def enable_compilation_cache(directory) -> None:
    """Persist compiled executables so a repeated program never recompiles.

    It pays on a resumed run, across workers sharing one directory, and -- with
    :func:`hoist_constants` -- across profiles on one window.
    """

    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(directory))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
