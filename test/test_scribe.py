# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


# import openai
# import os

# import prompttools.scribe  # noqa: F401 Importing this line will monkey-patch `openai.chat.completions.create`

# if __name__ == "__main__":
#
#     os.environ["OPENAI_API_KEY"] = ""
#
#     # Launch server from `app.py` first
#     # Example usage:
#     for i in range(3):
#         messages = [
#             {"role": "user", "content": f"What is 1 + {i}?"},
#         ]
#         result = openai.chat.completions.create(model="gpt-3.5-turbo", messages=messages, hegel_model="TEST_MODEL")
#         print(f"{i} {result = }")
#
#     print("End")
