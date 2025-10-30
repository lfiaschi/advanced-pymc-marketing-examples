# Development Guidelines

## Core Design Principles

### Architecture Philosophy
- **YAGNI (You Ain't Gonna Need It)**: Don't implement functionality until you really need it — avoid speculative features and premature complexity
- **DRY (Don't Repeat Yourself)**: Keep your codebase free of unnecessary repetition to facilitate maintainability and refactoring
- **Avoid Overengineering**: Build what is necessary, simplify solutions, and resist the urge to make code overly generic or abstract
- **Prefer functions over classes** — use classes only for true stateful or polymorphic code
- **Prioritize clarity, explicitness, and testability** across the codebase

### Functions and Methods
- **Keep functions short and do one thing**: Each function should have single, clear responsibility (high cohesion)
- Keep functions deterministic and single-purpose with clear inputs/outputs
- Avoid flags or hidden state — compose small functions for diverse logic
- Use short, expressive names with comprehensive docstrings

### Code Quality Standards
- **Use meaningful names**: Choose descriptive identifiers for variables, functions, and classes so the code is self-explanatory
- **Use _ naming conventions** for functions and variables
- **Use comments to document thought process, not just what the code does**: Add comments explaining reasoning or non-obvious choices in the code
- **Avoid hardcoded values**: Use constants and configuration files instead of burying magic numbers or values directly in code
- **Include type hints in every every function and variable declaration** for clarity and reliability
- Treat type hints as lightweight contracts
- Use defensive programming effectively: Do not over rely on try catch and do not catch generic exceptions!
- Validate all boundaries; keep core code assumption-safe

## Code Organization

### Module Structure
- **Organize yourself**: Structure your code with clarity, using folders and modules that reflect logical groupings
- **Don't overcomplicate module structure**: Avoid deep nesting or excessive splitting of files and modules; aim for balance between organization and simplicity
- Change code for one reason per module; keep modules short and focused
- Move reusable logic into modules; never hide critical transforms in notebook cells

### Code Evolution
- Reuse functionality when natural, not for backward compatibility with obsolete patterns
- Reimplement only if reuse causes debt or loss of clarity — never duplicate code per DRY principle
- Recommend improvements for optimization and readability where applicable

## Testing Strategy

### Test Philosophy
- **Write tests for critical code**: Focus testing on areas of the code that are complex or business-critical
- Write all tests as proper, maintainable test cases within the suite
- Even for quick logic verification, write real tests that offer ongoing value
- **Never create throwaway scripts** or ad hoc verification files

### Test Implementation
- Place tests in the `tests/` directory, mirroring the project's structure
- Ensure every test is runnable with `uv run pytest`
- Use functions within tests instead of classes, unless genuinely needed for state or polymorphism
- Maintain consistency across the test suite

## Tech Stack

### Data Processing
- **Use Polars for all data pipelines** — never default to Pandas but remember to convert to pandas when necessary
- Adopt vectorized operations for efficiency
- Favor `.with_columns()` chains and vectorized methods over loops
- Apply `.lazy()` for transformations on large datasets
- Prioritize vectorized operations over Python loops
- Avoid in-place mutation; prefer returning new objects
- Maintain explicit and stable column names — no positional magic

### Data Modeling
- For immutable, comparable value objects, use `@dataclass(frozen=True)`
- Ensure value-object methods remain pure and side-effect-free
- Only use regular classes for mutable state

## Development Environment and Tools

### Package Management
- **Use `uv` for all package installation and command execution**
- Never run Python or tool commands directly — always prefix with `uv run <command>`
- **Examples:**
  - `uv run python script.py` (not `python script.py`)
  - `uv run pytest` (not `pytest`)

### Style and Formatting
- Adhere strictly to **PEP8** standards
- Auto-format with **Black**, **ruff**, and **isort** after writing code
- Group all imports at the top of files — never within code blocks

## User Interface and Output

### Console Display Standards (DO NOT APPLY IN NOTEBOOKS)
- **Use the Rich library for all output** — avoid default `print()`
- Print with markup syntax for colors and emphasis (e.g., `[bold magenta]text[/bold magenta]`)
- Render tables and metrics with `Rich.table`
- Format structured content with `Console`