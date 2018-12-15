FROM nvidia/cuda:9.1-base

# Install vim, git, and cron
RUN apt-get update && apt-get -y install apt-file && apt-file update && apt-get -y install vim && \
    apt-get -y install cron && apt-get -y install git

#RUN wget https://www.python.org/ftp/python/3.7.1/Python-3.7.1.tgz

CMD /bin/bash