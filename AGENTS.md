## Testing approach

- Never create throwaway test scripts or ad hoc verification files
- If you need to test functionality, write a proper test in the test suite
- All tests go in the `tests/` directory following the project structure
- Tests should be runnable with the rest of the suite (`uv run pytest`)
- Even for quick verification, write it as a real test that provides ongoing value


## Package management

- This project uses uv for all package management
- Never run commands directly (python, pytest, etc.)
- Always prefix commands with `uv run <command>`
- Example: `uv run python script.py` not `python script.py`
- Example: `uv run pytest` not `pytest`


## Tech Stack 
- Always use **Polars** for new data pipelines not Pandas.  
- Apply lazy execution (`df.lazy()`) for transformations on large datasets.  
- Favor `.with_columns()` chains over Pandas-style mutations.  
- Validate datasets using **Pandera** schemas before modeling.
- Prefer vectorized Polars operations over loops.  

## Coding Standards
- Follow **PEP8** and auto-format with **Black**.  
- Type hints required in all functions.
- Include one Pytest test per core function in `/tests/`.

## Communication and Output
- All printed output should use the **Rich** library for styling and readability.
- Use `from rich import print` instead of Python’s built-in `print`.
- Apply markup syntax for color and emphasis (e.g., `[bold magenta]text[/bold magenta]`).
- Tabular summaries (e.g., DataFrames, metrics) rendered using `Rich.table`.  
- For structured or multiline objects, use the `Console` class.

## Default Agent Behaviors & Design Patterns

### 1. Core Principles
- Prefer **functions over classes**. Use a class only for real state or polymorphism.
- Write code that is **clear, explicit, and testable**. 
- Keep **calculation code pure**. Put I/O, logging, randomness, and plotting in thin wrappers.
- Use **short, meaningful filenames and function names** and **docstrings** that explain intent and assumptions.

### 2) Data Modeling
- Use `@dataclass(frozen=True)` for value objects that should be immutable and comparable.
- Keep dataclass methods pure. No network, file, or print calls.
- Use regular classes only when mutable state is essential.

### 3) Functions
- Small, deterministic, single-purpose. Explicit inputs and return values.
- No hidden state, globals, or implicit dependencies.
- Compose small functions rather than adding flags that switch behavior.

### 4) Typing and Contracts
- Add **type hints everywhere**. Treat them as lightweight contracts.
- Use `Protocol` or ABCs only when an interchangeable interface is truly needed.
- Validate external inputs at the boundary. Keep core code assumption-safe.

### 5) Reproducibility
- Set and pass **random seeds** from the top level. Do not call `seed` inside pure functions.
- Make runs deterministic when possible. If not, document sources of nondeterminism.

### 6) DataFrames and Arrays
- Prefer **vectorized operations** over Python loops.
- Avoid in-place mutation of shared data. Return new objects unless memory demands otherwise.
- Make column names explicit and stable. No magic positional indexing.

### 7) Style and Tooling
- Enforce `black`, `ruff`, and `isort`.
- One reason to change per module. Keep modules short and cohesive.
- Prefer expressive names over comments. Use docstrings for intent, invariants, and units.

### 9) Performance
- Optimize your code for clarity first.
- Use iterators or chunked processing only for large or streaming data.

### 10) Notebooks and Scripts
- Keep notebooks for exploration and reports. Move reusable logic into modules.
- Do not hide critical transforms in notebook cells. Library code lives in `src/`.

## 11) Code Evolution and Reuse
- Suggest optimization and readability improvements where relevant.  
- Reuse existing functionality when it fits naturally into the current feature or implementation.
- Do **not** maintain backward compatibility with obsolete patterns or interfaces — instead, rethink the design and refactor the implementation.
- Reimplement functionality **only** when reuse would introduce technical debt or reduce clarity — and never duplicate behavior.