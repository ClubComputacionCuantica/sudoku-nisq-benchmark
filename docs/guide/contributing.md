# Contributing and Doc Style

Thanks for contributing! This project uses Sphinx + MyST (Markdown) for pages and reStructuredText (reST) inside Python docstrings.

## Docstring style (important)
- Use reST in Python docstrings. Do NOT use triple backticks (`````); Sphinx parses reST.
- For code examples in docstrings, prefer::

  .. code-block:: python

      example_code()

- For attributes in class docstrings, use ``:ivar:`` to avoid duplicate autosummary targets.
- When referring to classes in other modules, disambiguate with fully qualified names, e.g. ``:class:`~sudoku_nisq.providers.base.QuantumProvider```.

## Pages (MyST)
- Write user guides in Markdown under ``docs/guide``.
- Use the existing toctree in ``docs/index.md`` and add links for new pages.

## Local checks
```bash
poetry run pytest
poetry run sphinx-build -b html docs docs/_build/html
```

```{todo}
Add a pre-commit hook to lint docstrings and prevent ``` fences in .py files.
```
