PHONY: isort rufflint ruffformat codeimprove

isort:
	isort ./

rufflint:
	ruff check ./ --fix

ruffformat:
	ruff format ./

codeimprove: ruffformat rufflint isort
