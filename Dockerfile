#FROM ubuntubase4:16.04
#FROM tensorflow/tensorflow:latest
FROM pymachine:latest

#COPY data/pickle/. /app   
COPY src/CNN/. /app   
COPY Makefile /app   
COPY model /app

WORKDIR /app
#RUN pip install -r requirements.txt
CMD ["make 3cnn"]