FROM python:3.7
ADD . /code
WORKDIR /code
RUN pip install -r requirements.txt
RUN unzip /code/movies_ds/archive.zip -d /code
CMD python app.py
