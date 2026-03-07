## Contributing to Ask Log

Thank you for considering a contribution! This document outlines how to propose changes and get your development environment set up.

### Code of Conduct

By participating, you agree to uphold a respectful and inclusive environment. Be kind and constructive.

### Getting Started

1. Fork and clone the repo.
2. Create a virtual environment and install in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows:
.\.venv\Scripts\Activate.ps1

pip install -U pip setuptools wheel
pip install -e .
```

3. Run tests:

```bash
pip install -r requirements.txt
pip install pytest
pytest -q
```

### Branching and Commits

- Create a feature branch: `feat/your-feature` or `fix/your-bug`.
- Keep commits small and focused. Use clear messages in imperative mood.

### Style

- Aim for clear, readable code with descriptive names.
- Prefer multi-line clarity over clever one-liners.

### Pull Requests

- Describe the problem and solution, with before/after where relevant.
- Add tests or adjust existing ones if behavior changes.
- Ensure the test suite is green.

### Issues and Discussions

- Use issues to report bugs or propose features. Provide logs, steps, and context.

### Releasing

- No PyPI release yet; install locally with `pip install -e .`.

Thanks for helping improve Ask Log!
