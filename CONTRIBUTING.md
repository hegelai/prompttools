# Contributing to `prompttools`

## TL;DR

We appreciate all contributions to our project! If you are interested in contributing to `prompttools`, there are many ways to help out.
Your contributions may fall into the following categories:

 It will greatly help our project if you:

- Star ⭐ our project and share it with your network!

- Report issues that you see, or upvote issues that others have reported and are relevant to you

- Look through existing issues for new feature ideas
  (["Help Wanted" issues](https://github.com/hegelai/prompttools/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22)) and open PRs to implement them.

- Answer questions on the issue tracker, investigating and fixing bugs are very valuable contributions to the project.

- Improve the documentation is welcomed. If you find a typo in the documentation,
  do not hesitate to submit a GitHub issue or pull request.

- Feature a usage example in our documentation, that is welcomed as well.

## Issues

We use GitHub issues to track bugs. Please follow the existing templates if possible and ensure that the
description is clear and has sufficient instructions to reproduce the issue.

You can also use open an issue to seek advice or discuss best practices of using our tool or prompting in general.

## Development installation

### Install `prompttools` from source

```bash
git clone https://github.com/hegelai/prompttools.git
cd prompttools
pip install -e .
pip install flake8
```

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
    - Optionally, you can create a new branch locally and push to the branch to `origin`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the inline documentation and examples.
4. Ensure all unit tests pass.
5. If you haven't already, complete the Contributor License Agreement ("CLA"). More details below

### Code style

`prompttools` adheres to code format through [`pre-commit`](https://pre-commit.com). You can install it with

```shell
pip install pre-commit
```

To check and in most cases fix the code format, stage all your changes (`git add`) and run `pre-commit run`.

We recommend you to perform the checks automatically before every `git commit`, you can install that by executing
this in the directory:

```shell
pre-commit install
```


## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to sign a CLA. You only need to do this once to work on our project.

Please sign the CLA here: <https://cla-assistant.io/hegelai/prompttools>

## License

By contributing to `prompttools`, you agree that your contributions will be licensed under the LICENSE file in the root
directory of this source tree.
