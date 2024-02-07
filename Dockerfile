FROM ubuntu:22.04

WORKDIR /sci_tf-model

RUN apt update 
RUN apt -y upgrade 
RUN apt install -y pip

COPY requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

COPY sci_tf ./sci_tf
COPY train_main.py ./train_main.py

CMD ["python3", "sci_tf-model/train_main.py"]
