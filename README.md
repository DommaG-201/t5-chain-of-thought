First, run in terminal:

pip install simplet5 -q

pip install sklearn -q

pip install datasets -q

pip install pandas -q

terminal commands for running in terminal w/ docker imgs:

`cd dissertation`

`hare build -t dg707/disseration .`

`hare run --rm --gpus device=1 --workdir /app -v "$(pwd)":/app --user $(id -u):$(id -g) dg707/dissertation:latest  python3 -x main.py`