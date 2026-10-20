"""Hoisted constants: one program for every value of what a function closes over."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tellurix.compiled import hoist_constants


def _closure(seed):
    table = jnp.asarray(np.random.default_rng(seed).random(1000))
    return lambda x, extra: {"sum": jnp.sum(jnp.sin(x * table)) + extra["shift"], "table": table * x}


def test_two_closures_lower_to_one_program_and_keep_their_own_values():
    example = (2.0, {"shift": 1.0})
    programs, results = [], []
    for seed in (1, 2):
        fun = _closure(seed)
        run, constants = hoist_constants(fun, *example)
        compiled = jax.jit(run)
        programs.append(compiled.lower(constants, *example).as_text())
        results.append((compiled(constants, *example), fun(*example)))
    # The program the persistent cache keys on is the same; jit of the closure
    # itself would bake each table in.
    assert programs[0] == programs[1]
    assert jax.jit(_closure(1)).lower(*example).as_text() != jax.jit(_closure(2)).lower(*example).as_text()
    for hoisted, direct in results:
        assert float(hoisted["sum"]) == pytest.approx(float(direct["sum"]), rel=1e-15)
        np.testing.assert_array_equal(hoisted["table"], direct["table"])
