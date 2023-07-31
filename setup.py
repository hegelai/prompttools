#!/usr/bin/env python

import distutils.command.clean
import os
import shutil
import subprocess

from pathlib import Path

from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).parent.resolve()


def _get_requirements():
    """Get dependency requirements from `requirements.txt`."""
    req_list = []
    with Path("requirements.txt").open("r") as f:
        for line in f:
            req = line.strip()
            if len(req) == 0 or req.startswith("#"):
                continue
            req_list.append(req)
    return req_list


def _get_version():
    """Get package version."""
    with open(os.path.join(ROOT_DIR, "version.txt")) as f:
        version = f.readline().strip()

    sha = "Unknown"
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT_DIR)).decode("ascii").strip()
    except Exception:
        pass

    os_build_version = os.getenv("BUILD_VERSION")
    if os_build_version:
        version = os_build_version
    elif sha != "Unknown":
        version += "+" + sha[:7]

    return version, sha


def _export_version(version, sha):
    version_path = ROOT_DIR / "prompttools" / "version.py"
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = {repr(sha)}\n")


requirements = _get_requirements()


class Clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove prompttools extension
        def remove_extension(pattern):
            for path in (ROOT_DIR / "prompttools").glob(pattern):
                print(f"removing extension '{path}'")
                path.unlink()

        for ext in ["so", "dylib", "pyd"]:
            remove_extension("**/*." + ext)

        # Remove build directory
        build_dirs = [
            ROOT_DIR / "build",  # Remove build
            ROOT_DIR / "prompttools.egg-info",  # Remove egg metadata
        ]
        for path in build_dirs:
            if path.exists():
                print(f"removing '{path}' (and everything under it)")
                shutil.rmtree(str(path), ignore_errors=True)


if __name__ == "__main__":
    VERSION, SHA = _get_version()
    # TODO: Exporting the version here breaks `python -m build`
    _export_version(VERSION, SHA)

    print("-- Building version " + VERSION)

    setup(
        # Metadata
        name="prompttools",
        version=VERSION,
        description="Tools for prompts.",
        long_description=Path("README.md").read_text(encoding="utf-8"),
        long_description_content_type="text/markdown",
        url="https://github.com/hegelai/prompttools",
        author="Hegel AI",
        author_email="steve@hegel-ai.com, kevin@hegel-ai.com",
        license="Proprietary",
        install_requires=requirements,
        python_requires=">=3.10",
        classifiers=[
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        # Package Info
        packages=find_packages(exclude=["test*", "examples*", "build*"]),
        zip_safe=False,
        cmdclass={
            "clean": Clean,
        },
    )
