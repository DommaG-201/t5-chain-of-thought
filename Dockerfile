FROM python:3.8
WORKDIR /app
RUN pip3 install simplet5 -q && install sklearn -q && pip3 install datasets -q && pip3 install pandas -q
COPY . .
CMD ["python3", "main.py"]