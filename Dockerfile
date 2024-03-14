FROM python:3.11

RUN apt update
RUN apt install libmpc-dev -y

ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

ADD . /ts_robustness
RUN pip install -r ts_robustness/requirements.txt

WORKDIR /ts_robustness
CMD [ "/bin/bash" ]