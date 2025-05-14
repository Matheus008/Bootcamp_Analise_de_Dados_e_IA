FROM python:3.13.2

WORKDIR /src

COPY . .

RUN pip install --no-cache-dir -r dependencias.txt

CMD ["python", "src/main.py"]