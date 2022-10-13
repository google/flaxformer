# Flaxformer: transformer architectures in JAX/Flax

Flaxformer is a transformer library for primarily NLP and multimodal research at
Google. It is used for many NLP research use cases, providing both off-the-shelf
BERT and T5 models, and several research projects built on shared components.

## General library goals

The Flaxformer library aims to provide transformer models that are:

*   **High performance**: Models are annotated for use with the PJIT API,
    enabling them to be used for training the largest models.
*   **Reusable**: Components have self-contained configuration, and high-level
    modules like encoders, decoders, etc. don't make too many assumptions about
    what their sub-modules look like.
*   **Tested**: We aim to employ a reasonable amount of unit testing, and write
    tests whenever bugs are encountered. However no guarantees are provided.
*   **Maintainble**: We have created a versioning strategy for our modules so
    code refactors can take place which alter the module structure. This is
    tricky in Flax, because Flax generates a tree of parameters based on the
    exact module structure. Our approach lets us maintain compatibility with
    previously trained model checkpoints.

## Code locations

Modeling components such as dense attention, layer norms, and MLP blocks can be
found in the `components/` directory.

Higher-level classes which combine these components can be found in the
`architectures/` directory. The current architecture file for the T5 family of
models is `architectures/t5/t5_architecture.py`; this is a mid-level API
requiring sub-components to be configured. A high-level starting point, exposing
fewer parameters, is `architectures/t5/t5_1_1.py`.

## Relationship to other codebases

Flaxformer is primarily used by other research projects, in particular
[T5X](https://github.com/google-research/google-research/tree/master/flax_models/t5x).
We hope to release examples demonstrating the integration of these codebases
soon.

If you would like to use Flaxformer independently of T5X, please see the unit
tests for examples instantiating the models. In the medium-term future, we hope
to provide more stand-alone examples of Flaxformer use.

## Contributions

Unfortunately, we cannot accept contributions to the Flaxformer repo at this
time, so any pull requests will be automatically closed - but please file issues
as needed!

# Installing dependencies and running tests

First, we recommend installing a few dependencies manually,

```
pip3 install numpy sentencepiece tensorflow==2.8.1
```

This is a workaround to prevent pip backtracking on package versions; we
believe there is either a version conflict in upstream packages, or pip's
constraint solving process is imperfect.

Then, check out this repository. In its root directory, you can install it
along with test dependencies by running,

```
pip3 install '.[testing]'
```

If you like, you can run the tests from pytest with the following invocation,

```
python3 -m pytest
```

## Uninstalling

If you need to uninstall Flaxformer, please run,

```
pip3 uninstall flaxformer
```

## Troubleshooting

### Flax deps

Flaxformer is developed in close collaboration with the Flax team. There may be
bugs if your Flax version is not up to date. To install the latest version from
GitHub, please run,

```
pip3 uninstall flax
pip3 install git+https://github.com/google/flax
```

## Note

Flaxformer is a project maintained by a team in Google Research. It is not an
official Google product.

