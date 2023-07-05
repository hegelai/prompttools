def is_interactive() -> bool:
    """
    Used to determine if we are in a jupyter notebook, which
    determines how we present the visualizations.
    """
    import __main__ as main

    return not hasattr(main, "__file__")
