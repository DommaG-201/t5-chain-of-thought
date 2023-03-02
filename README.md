First, run in terminal:

`pip install simplet5 -q`

`pip install sklearn -q`

`pip install datasets -q`

`pip install pandas -q`

to install the dependancies. Or, run with docker.

terminal commands for running in terminal w/ docker imgs:

`hare build -t <username>/<imgname> .`

`hare run --rm --gpus device=1 --workdir /app -v "$(pwd)":/app --user $(id -u):$(id -g) <username>/<imgname>:latest  python3 -x main.py`

Note that when running in terminal you can adjust the program as following:

`python main.py <T5_MODEL_NAME> <ADJUST_QUESTIONS (True or False)> <MAX_VALUE (int)> <MIN_VALUE (int)>`

If any variable isn't defined, then the var will default to the following values:

T5_MODEL_NAME = t5-base

ADJUST_QUESTIONS = False

MAX_VALUE = 1000

MIN_VALUE = 1
