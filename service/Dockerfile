FROM python:3.6

# Install vim, git, and cron
RUN apt-get update && apt-get -y install apt-file && apt-file update && \
    apt-get -y install cron git gcc vim

# place to keep our app and the data:
RUN mkdir -p /app && \
    mkdir -p /app/models && \
    mkdir -p /app/logs && \
    mkdir -p /data

## Add crontab file in the cron directory
#ADD code/crontab /etc/cron.d/fetch-cron
## Give execution rights on the cron job
#RUN chmod 0644 /etc/cron.d/fetch-cron
## Apply cron job
#RUN crontab /etc/cron.d/fetch-cron
## Create the log file to be able to run tail
#RUN touch /var/log/cron.log

# install python libs
#RUN pip install Cython && pip install numpy
COPY code/requirements.txt /app/
RUN pip install numpy>=1.15.1 && pip install -r /app/requirements.txt

# copy over the secrets:
COPY secrets.json /app/

# copy over the code
ADD code/ /app/

# copy over the models
ADD models/ /app/models/

# change working directory to /app
WORKDIR /app

#CMD /bin/bash
# run fetcher only
CMD /usr/local/bin/supervisord -n -c supervisord.fetcher.conf

#CMD cron && crontab /etc/cron.d/fetch-cron && /bin/bash
#CMD cron && crontab /etc/cron.d/fetch-cron && /usr/local/bin/supervisord -n -c supervisord.conf
#CMD gunicorn -w 4 -b 0.0.0.0:4000 server:app
