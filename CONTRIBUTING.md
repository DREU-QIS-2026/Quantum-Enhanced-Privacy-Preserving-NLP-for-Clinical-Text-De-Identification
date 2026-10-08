# Contributing to QuantumHPC

Thank you for contributing to the QuantumHPC research project!

This repository supports a collaborative DREU research project. Please follow the guidelines below to keep the repository organized and maintain a stable codebase.

---

# Development Workflow

Never develop directly on the **main** branch.

Instead:

1. Update your local repository.
2. Switch to your development branch.
3. Make your changes.
4. Test your code.
5. Commit your changes.
6. Push your branch.
7. Open a Pull Request.

---

# Branches

Current development branches

| Branch | Purpose | Maintainer |
|---------|----------|------------|
| main | Stable project version | Kyle |
| kyle-hpc | HPC, benchmarking, MPI | Kyle |
| edmund-quantum | Quantum circuits & algorithms | Edmund |
| sofia-clinical | Clinical datasets & preprocessing | Sofia |

---

# Pull Before Working

Always begin by updating your local repository.

```bash
git checkout main
git pull origin main
```

Then switch to your development branch.

```bash
git checkout YOUR_BRANCH
```

Merge the latest changes.

```bash
git merge main
```

---

# Commit Messages

Write clear commit messages.

Good examples

```text
feat: add MPI driver

feat: improve Aer simulator logging

fix: correct dataset loader

docs: update TEAM_SETUP guide

refactor: simplify circuit builder

test: add benchmark validation
```

Avoid messages like

```text
update

changes

fixed stuff

asdf
```

---

# Before Committing

Please verify:

- Code runs successfully.
- No unnecessary files are included.
- Imports are working.
- Temporary files are removed.
- Virtual environment is NOT committed.

Run

```bash
git status
```

before every commit.

---

# Files That Should Never Be Committed

Do NOT commit

```text
.venv/

__pycache__/

.vscode/

*.pyc

Release/

*.zip
```

Git should already ignore these files.

---

# Pull Requests

Every Pull Request should include

- Summary of changes
- Reason for changes
- Files modified
- Any known issues

Example

```text
Summary

Implemented MPI driver.

Reason

Preparing distributed benchmarking.

Files

mpi_driver.py

benchmark.py
```

---

# Code Style

General guidelines

- Use meaningful variable names.
- Write comments where necessary.
- Keep functions focused on one task.
- Prefer readability over clever code.

---

# Repository Organization

Current project structure

```
backends/
benchmark/
circuits/
datasets/
data/
docs/
logs/
mpi/
plots/
results/
simulation/
tests/
```

Keep files within their appropriate folders.

---

# Team Responsibilities

Kyle

- Repository maintenance
- HPC
- Benchmarking
- MPI
- Performance analysis

Edmund

- Quantum circuits
- Feature maps
- Algorithm implementation

Sofia

- Clinical datasets
- Data preprocessing
- NLP integration

---

# Research Log

Each week, update the DREU research log.

Location

```
logs/
```

Each weekly entry should include

- Goals
- Approach & Implementation
- Results
- Next Steps

---

# Questions

If you are unsure how to proceed,

Ask before pushing.

It is much easier to prevent a problem than to fix one after it reaches the main branch.
