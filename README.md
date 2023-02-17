First, run in terminal:

`pip install simplet5 -q`

`pip install sklearn -q`

`pip install datasets -q`

`pip install pandas -q`

to install the dependancies. Or, run with docker.

terminal commands for running in terminal w/ docker imgs:


`hare build -t <username>/<imgname> .`

`hare run --rm --gpus device=1 --workdir /app -v "$(pwd)":/app --user $(id -u):$(id -g) <username>/<imgname>:latest  python3 -x main.py`
