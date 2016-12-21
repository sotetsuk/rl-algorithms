.PHONY: clean build

clean:
	rm -rf build
	rm -rf dist
	rm -rf rl_algorithms.egg-info

build:
	python setup.py install
