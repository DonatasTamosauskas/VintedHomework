FROM python:3.6-stretch

# Set the default shell to bash rather than sh
ENV SHELL /bin/bash

RUN pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
RUN pip install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl
RUN pip install fastai
RUN pip install flask
RUN pip install Pillow

RUN mkdir ClothingClassifier
WORKDIR /ClothingClassifier

# Expose the flask port outside of docker
EXPOSE 5000

# Lanch the flask app
COPY . .
CMD "flask run --host=0.0.0.0"
