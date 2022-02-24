PYTHON ?= python3

.PHONY: test check example verify

test:
	$(PYTHON) -m unittest discover -s tests -v

check:
	$(PYTHON) -m compileall -q inertialnav tests examples
	git diff --check

example:
	$(PYTHON) -m examples.simulate
	$(PYTHON) -m examples.replay_csv examples/level_imu.csv

verify: test check example
