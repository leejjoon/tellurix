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


@pytest.mark.parametrize("given, expected", [
    ("", "--xla_cpu_experimental_ynn_fusion_type="),
    ("--xla_force_host_platform_device_count=2",
     "--xla_force_host_platform_device_count=2 --xla_cpu_experimental_ynn_fusion_type="),
    ("--xla_dump_to=/dev/null --xla_cpu_experimental_ynn_fusion_type=",
     "--xla_dump_to=/dev/null --xla_cpu_experimental_ynn_fusion_type="),
])
def test_import_turns_off_ynn_fusion_unless_the_caller_chose(given, expected):
    # In a fresh interpreter: XLA reads the flag only when the backend starts,
    # and the flag must be one this jaxlib knows, or the computation aborts.
    import os
    import subprocess
    import sys

    code = ("import os, tellurix, jax.numpy as jnp; float(jnp.ones(3).sum()); "
            "print(os.environ['XLA_FLAGS'])")
    env = {**os.environ, "XLA_FLAGS": given, "JAX_PLATFORMS": "cpu"}
    output = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                            check=True).stdout
    assert output.strip() == expected
