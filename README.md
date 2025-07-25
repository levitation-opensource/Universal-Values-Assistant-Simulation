

## Project setup

This readme contains instructions for both Linux and Windows installation. Windows installation instructions are located after Linux installation instructions.

### Installation under Linux

To setup the environment follow these steps:

1. Install CPython. The code is tested with Python version 3.10.10. We do not recommend using Conda package manager. 

Under Linux, run the following commands:

`sudo add-apt-repository ppa:deadsnakes/ppa`
<br>`sudo apt update`
<br>`sudo apt install python3.10 python3.10-dev python3.10-venv`
<br>`sudo apt install curl`
<br>`sudo curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10`

2. Get the code from repo:

`sudo apt install git-all`
<br>Run `git clone https://github.com/levitation-opensource/Universal-Values-Assistant-Simulation.git`
<br>Run `cd Universal-Values-Assistant-Simulation`

3. Create a virtual python environment:

`python3.10 -m venv_UVAS`
<br>`source venv_UVAS/bin/activate`

4. Install dependencies by running:
<br>`pip install -r requirements.txt`


### Installation under Windows

1. Install CPython from python.org. The code is tested with Python version 3.10.10. We do not recommend using Conda package manager.

You can download the latest installer from https://www.python.org/downloads/release/python-31010/ or if you want to download a newer 3.10.x version then from https://github.com/adang1345/PythonWindows

2. Get the code from repo:
* Install Git from https://gitforwindows.org/
* Open command prompt and navigate top the folder you want to use for repo
* Run `git clone https://github.com/levitation-opensource/Universal-Values-Assistant-Simulation.git`
* Run `cd Universal-Values-Assistant-Simulation`

3. Create a virtual python environment by running: 
<br>3.1. To activate VirtualEnv with Python 3.10:
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`virtualenv -p python3.10 venv_UVAS` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(or if you want to use your default Python version: 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`python -m venv venv_UVAS`)
<br>3.2. `venv_UVAS\scripts\activate`

4. Install dependencies by running:
<br>`pip install -r requirements.txt`


## Executing `Universal Values Assistant Simulation`

Choose model in `config.ini`.

Set environment variable:
`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.

Ensure you have loaded credits on your corresponding LLM API account, else you will get "rate limit errors".

Run 
<br>`python Simulation.py`


# License

This project is licensed under the Mozilla Public License 2.0. You are free to use, modify, and distribute this code under the terms of this license.

For more details, see the [LICENSE.txt](LICENSE.txt) file.
