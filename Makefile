.PHONY: clean-pyc clean-build docs clean
BROWSER := python -mwebbrowser

help:
	@echo "clean - remove all build, test, coverage and Python artifacts"
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-test - remove test and coverage artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "test-all - run tests on every Python version with tox"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "release - package and upload a release"
	@echo "dist - package"
	@echo "install - install the package to the active Python's site-packages"
	@echo "develop - install the package in development mode"

clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -f coverage.xml
	rm -fr htmlcov/

lint:
	flake8 skmisc --exclude=skmisc/__config__.py

build:
	./dev.py build

test: clean-test
	./dev.py test

sdist: clean-test
	./dev.py sdist

coverage:
	./dev.py coverage
	./dev.py coverage-html
	$(BROWSER) htmlcov/index.html

docs:
	./dev.py docs
	$(BROWSER) doc/_build/html/index.html

release: clean
	bash ./tools/release.sh

dist: clean
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean
	python setup.py install

develop: build
	pip install --no-build-isolation .
