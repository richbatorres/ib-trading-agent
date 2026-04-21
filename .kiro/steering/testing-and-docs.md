# Testing, Test Execution & Documentation Rules

inclusion: always

## Test Execution Method (Windows / Google Drive)

This project lives on Google Drive which causes shell blocking issues. ALWAYS run tests using this exact pattern:

```
# 1. Copy updated files from workspace to local temp
C:\temp\trading-agent\venv\Scripts\python.exe -c "
import shutil, os
src = 'G:/My Drive/Documents/practice/trading agent'
dst = 'C:/temp/trading-agent'
files = ['src/...', 'tests/unit/...']  # list changed files
for f in files:
    shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
print('COPIED')
"

# 2. Run tests via background process with subprocess + timeout
controlPwshProcess start:
$env:PYTHONPATH="C:\temp\trading-agent"
C:\temp\trading-agent\venv\Scripts\python.exe -c "
import subprocess, sys
r = subprocess.run(
    [sys.executable, '-m', 'pytest', <test_files>, '-v', '--tb=line',
     '--no-header', '-p', 'no:cacheprovider'],
    cwd=r'C:\temp\trading-agent',
    capture_output=True, text=True, timeout=90
)
f = open(r'C:\temp\pytest_out.txt', 'w')
f.write(r.stdout + '\n' + r.stderr)
f.close()
print('DONE')
"
while($true){Start-Sleep 1}

# 3. Read results via getProcessOutput or separate terminal
```

**NEVER** use `executePwsh` directly for pytest — it blocks the shell.
**ALWAYS** use `controlPwshProcess` with the subprocess wrapper pattern above.
**ALWAYS** append `; while($true){Start-Sleep 1}` to keep the background process alive.

## Mandatory Testing for New Features

When adding ANY new functionality:

1. **Write tests FIRST or alongside the implementation** — never ship code without tests.
2. **Test file location**: Unit tests go in `tests/unit/test_<module>.py`. Integration tests in `test/test_agent.py`. Performance tests in `test/test_performance.py`.
3. **Test naming**: Use descriptive names like `test_<what>_<condition>_<expected>`.
4. **Determinism**: All tests MUST use fixed seeds (`np.random.default_rng(42)`) and mock all external APIs (IB, SMTP, Polymarket, Anthropic).
5. **Run tests after every significant code change** — at minimum after completing a feature or fixing a bug.
6. **Fix failing tests immediately** — never leave broken tests for later.

## Mock Patterns (Lessons Learned)

- Use `[ticker]` (list) not `{ticker}` (set) for `on_pending_tickers` — `SimpleNamespace` is not hashable.
- For `ib_insync` event `+=` mocking: the `__iadd__` callback receives `(self, cb)` not just `(cb)`.
- When mocking `datetime`, use `wraps=real_datetime` to preserve `strptime` — don't mock the entire class.
- For logging tests, filter by our specific handler types (stdout StreamHandler, TimedRotatingFileHandler) — pytest adds its own LogCaptureHandler.
- Async tests that use `asyncio.sleep` can block pytest — use short timeouts or skip in CI.
- When running ALL tests together, async tests (shutdown_handler, strategy_engine, test_agent) can interfere with sync tests that use `_run()` helper. Run them in two groups:
  - Group 1 (sync): all tests EXCEPT test_shutdown_handler.py, test_strategy_engine.py, test_agent.py
  - Group 2 (async): test_shutdown_handler.py, test_strategy_engine.py, test_agent.py

## Documentation Updates

After every significant code change, update the relevant documentation:

1. **requirements.md** — Update if new requirements are added or existing ones change.
2. **design.md** — Update if architecture, components, interfaces, data models, or correctness properties change.
3. **tasks.md** — Update task status, add new tasks if scope expands.
4. **README.md** — Update if setup instructions, CLI commands, or dependencies change.
5. **docs/deployment.md** — Update if deployment configuration changes.
6. **Steering files** — Update if conventions, rules, or patterns change.

The goal: anyone (including future me) should be able to read the .md files and understand the current state of the project without reading all the code.

## Code Hygiene

After every significant change:

1. **Remove temp files** — delete any scripts, batch files, or output files created during debugging.
2. **Check for unused imports** — scan modified files for imports that are no longer used.
3. **Remove dead code** — delete commented-out code, unused functions, or unreachable branches.
4. **Keep files focused** — if a module grows beyond ~300 lines, consider splitting it.
5. **Consistent style** — follow the existing code style (docstrings, type hints, logging patterns).

## Backup

After every significant session or before ending work:

1. **Run backup** to `G:\My Drive\Documents\practice\kiro-backup\`:
   - `.kiro/` folder (steering, specs, settings)
   - `.env` and `.env.example`
   - `docs/` folder
   - `pyproject.toml` and `README.md`
2. **Use this script pattern**:
```python
import shutil, os
src = 'G:/My Drive/Documents/practice/trading agent'
dst = 'G:/My Drive/Documents/practice/kiro-backup'
os.makedirs(dst, exist_ok=True)
kiro_dst = os.path.join(dst, '.kiro')
if os.path.exists(kiro_dst): shutil.rmtree(kiro_dst)
shutil.copytree(os.path.join(src, '.kiro'), kiro_dst)
docs_dst = os.path.join(dst, 'docs')
if os.path.exists(docs_dst): shutil.rmtree(docs_dst)
shutil.copytree(os.path.join(src, 'docs'), docs_dst)
for f in ['.env', '.env.example', 'pyproject.toml', 'README.md']:
    p = os.path.join(src, f)
    if os.path.exists(p): shutil.copy2(p, os.path.join(dst, f))
```
3. **When to backup**: after adding new features, fixing bugs, changing config, or before ending a session.
