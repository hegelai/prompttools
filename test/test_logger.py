# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


if False:  # Skipping this in CI

    import openai
    import prompttools.logger  # noqa: F401 Importing this line will monkey-patch `openai.chat.completions.create`


r"""
Example of using `prompttools.logger`.

All you need to do is call `import prompttools.logger` to start logging.
You can optionally add `hegel_model` to your call (as seen below). This will associate
this call with a specific name in the logs.

The OpenAI call is unchanged, it executes normally between your machine and OpenAI's server.

Note:
You should have "HEGELAI_API_KEY" and "OPENAI_API_KEY" loaded into `os.environ`.
"""

if __name__ == "__main__":
    if False:  # Skipping this in CI
        for i in range(1):
            messages = [
                {"role": "user", "content": f"What is 1 + {i}?"},
            ]

            # `hegel_model` is an optional argument that allows you to tag your call with a specific name
            # Logging still works without this argument
            # The rest of the OpenAI call happens as normal between your machine and OpenAI's server
            openai_response = openai.chat.completions.create(
                model="gpt-3.5-turbo", messages=messages, hegel_model="Math Model"
            )
            print(f"{openai_response = }")

        print("End")
