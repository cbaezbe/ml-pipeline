install:
	pip install -r requirements.txt

train:
	python src/train.py

test:
	pytest tests/