.PHONY: install run build

install:
	pip install -r requirements.txt

run:
	streamlit run approval_app.py

build:
	docker compose build --no-cache
	docker compose up -d --force-recreate

stop:
	docker compose down