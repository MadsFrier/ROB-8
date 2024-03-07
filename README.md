# ROB-8
abstact

## Installation Guide

On your host pc, run the <code>bash_requirements.sh</code> script to install needed packages and dependencies for the docker file to be able to use runtime nvidia.

Git clone the repository and cd into the docker folder

Build the dockerfile using vscode devcontainer extension (the docker container will be approx. 50 Gb so it will take a while)

In the docker container, follow the instructions on the <code>init.txt</code> file.

in */workspace/ROB-8/docker* directory, run 

<code>$ python3 src/demo/lseg_demo.py</code>

to run the demo for lseg segmentation on an image.