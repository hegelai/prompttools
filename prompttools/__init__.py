# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import sentry_sdk

sentry_sdk.init(
  dsn="https://43fbb5a3a556ca0a879f5a08ce805d87@o4505656408211456.ingest.sentry.io/4505656412667904",

  # Set traces_sample_rate to 1.0 to capture 100%
  # of transactions for performance monitoring.
  # We recommend adjusting this value in production.
  traces_sample_rate=1.0,
  debug=True
)

# TODO: Remove before merging. 
# This is to test that we can send errors to sentry
division_by_zero = 1 / 0

from .prompttest import prompttest


__all__ = ["prompttest"]
