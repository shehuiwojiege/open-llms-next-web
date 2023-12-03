FROM wallies/python-cuda:3.10-cuda11.7-runtime
ENV PYTHONUNBUFFERED 1
RUN mkdir /open-llms-next-web
WORKDIR /open-llms-next-web
COPY . /open-llms-next-web/
RUN pip install -r req.txt --default-timeout=5000  -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
