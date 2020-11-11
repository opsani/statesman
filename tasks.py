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

@invoke.task()
def test(task):
    task.run("poetry run pytest --cov=statesman --cov-report=term-missing:skip-covered --cov-config=setup.cfg .")

@invoke.task()
def typecheck(task):
	task.run("poetry run mypy . || true")

@invoke.task(name="lint-docs")
def lint_docs(task):
	task.run("poetry run flake8-markdown \"**/*.md\" || true")

@invoke.task(lint_docs)
def lint(task):
	task.run("poetry run flakehell lint --count")

@invoke.task(name="pre-commit")
def pre_commit(task):
    task.run("poetry run pre-commit install")
    task.run("poetry run pre-commit run --all-files")
