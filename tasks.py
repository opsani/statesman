import invoke


@invoke.task(name="format")
def _format(task):
    task.run("poetry run isort .")
    task.run("""poetry run autoflake --recursive \
        --ignore-init-module-imports \
        --remove-all-unused-imports  \
        --remove-unused-variables    \
        --in-place ."""
    )
    task.run("autopep8 --in-place --aggressive --aggressive statesman.py")


@invoke.task()
def test(task):
    task.run("poetry run pytest --cov=statesman --cov-report=term-missing:skip-covered --cov-config=setup.cfg .", pty=True)


@invoke.task()
def typecheck(task):
    task.run("poetry run mypy . || true")


@invoke.task(name="lint-docs")
def lint_docs(task):
    task.run("poetry run flake8-markdown \"**/*.md\" || true", pty=True)


@invoke.task(lint_docs)
def lint(task):
    task.run("poetry run flakehell lint --count", pty=True)


@invoke.task(name="pre-commit")
def pre_commit(task):
    task.run("poetry run pre-commit install", pty=True)
    task.run("poetry run pre-commit run --all-files", pty=True)
