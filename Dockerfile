FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

RUN chmod 777 /tmp && apt-get update
RUN apt-get install python3.10 -y
RUN apt-get install python-is-python3 -y
RUN apt-get install pip -y
RUN mkdir /open-llms-next-web
WORKDIR /open-llms-next-web
ADD req.txt /open-llms-next-web/
RUN pip install --no-cache-dir -r req.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
