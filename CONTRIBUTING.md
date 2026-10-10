# Contributing to AlphaReq

Thank you for your interest in contributing to **AlphaReq**! We welcome contributions from software engineering researchers, systems architects, and open-source developers.

---

## 1. Code of Conduct
We are committed to providing a friendly, safe, and welcoming environment for all contributors, regardless of experience level, background, or identity.

---

## 2. Development Setup

### Clone and Install
```bash
git clone https://github.com/umertanveer25/AlphaReq.git
cd AlphaReq

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

## 3. Running the Test Suite
Before submitting any pull request, ensure all 19 automated tests pass cleanly:
```bash
python -m pytest tests/ -v
```

All new features or bug fixes must be accompanied by relevant unit tests in `tests/`.

---

## 4. Coding Standards
- Adhere to **PEP 8** style guidelines.
- Use explicit type annotations on public function/method signatures.
- Add structured docstrings explaining arguments and return values.
- Avoid introducing unseeded randomness; always bind stochastic routines to `RandomState(seed)` for bitwise reproducibility.

---

## 5. Pull Request Workflow
1. Fork the repository and create a descriptive feature branch (`git checkout -b feature/dynamic-enclaves`).
2. Implement your changes with corresponding tests.
3. Run the full pipeline to verify that benchmark metrics and figures remain consistent:
   ```bash
   python run_pipeline.py
   ```
4. Commit your changes with concise, informative commit messages.
5. Push to your branch and submit a Pull Request to `main`.
