FROM python:3.6

# @TODO to be replaced with `pip install pommerman -i http://pypi.douban.com/simple/ --trusted-host mirrors.aliyun.com`
ADD . /pommerman
RUN pip install -U pip
RUN pip config set global.index-url http://mirrors.aliyun.com/pypi/simple
RUN pip config set install.trusted-host mirrors.aliyun.com
RUN cd /pommerman && pip install . 
# end @TODO

EXPOSE 10080

ENV NAME Agent

# Run app.py when the container launches
ADD ./examples/docker-agent /agent
WORKDIR /agent
ENTRYPOINT ["python"]
CMD ["run.py"]
