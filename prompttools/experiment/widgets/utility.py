# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


def is_interactive() -> bool:
    r"""
    Used to determine if we are in a jupyter notebook, which
    determines how we present the visualizations.
    """
    import __main__ as main

    return not hasattr(main, "__file__")
