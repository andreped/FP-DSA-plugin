# setup docker image from Python image
FROM python:3.9-slim

# install histomicstk necessary for communicating with HistomicsUI
RUN python -m pip install histomicstk --find-links https://girder.github.io/large_image_wheels

# set working directory and setup environment
ADD . /opt/FPweb/dsa
WORKDIR /opt/FPweb/dsa/cli
ENV PYTHONPATH=$PYTHONPATH:/opt/FPweb/dsa

# test entrypoint
RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli
RUN python -m slicer_cli_web.cli_list_entrypoint FPInference --help

# define docker entrypoint
ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]
