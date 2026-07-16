# NEP_CPU numerical reference fixtures

These fixtures are the minimal ordinary NEP and qNEP numerical tests copied
from the sibling NEP_CPU repository at commit
`9977dab39ae7ec23d64a73a1788815002824ace4`. The source fixture directories
were clean when copied.

Each case contains only the model, periodic `xyz.in` structure, and committed
force, per-atom raw9 virial, and descriptor references. Generated outputs,
finite-difference data, GPU outputs, source code, and build files are excluded.

The `nep` case is a 250-atom ordinary NEP3 model. The `qnep` case is a
250-atom `nep4_charge1` model. Both are exercised through the public `cpu`
backend; CUDA tests reuse the qNEP fixture when CUDA is enabled.
