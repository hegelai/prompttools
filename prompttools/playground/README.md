## `prompttools` Playground

If you are interested to have experiment with a UI rather than a notebook, the playground allows you to do that!
You can:
- Evaluate different instructions (system prompts)
- Try different prompt templates
- Compare across models (e.g. GPT-4 vs. local LLaMA 2)

<p align="center">
  <img src="../../img/playground.gif" width="1000" height="500">
</p>

To launch the playground locally, clone the git repo and run the following script with streamlit:

```
git clone https://github.com/hegelai/prompttools.git
cd prompttools && pip install -r prompttools/playground/requirements.txt
streamlit run prompttools/playground/playground.py
```

Similar to the notebook examples, all the executions and calls to LLM services happen within your local machines,
`prompttools` do not forward your requests or log your information.
