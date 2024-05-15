# Include the base image for the docker
# You can use an image based on PyTorch, Tensorflow, MXNet etc, depending on your prefered machine learning tool.
FROM pytorch/pytorch


# Setting working directory to /opt, rather than doing all the work in root.
# Copying the /code directory into /opt
WORKDIR /opt
COPY ./src /opt 

# Running pip install to download required packages
RUN pip install -r requirements.txt

# Setting the default code to run when a container is launced with this image.
# Comment the ENTRYPOINT line below if you want to run the container in interactive mode and run the scripts manually.
ENTRYPOINT [ "/bin/bash", "/opt/run.sh"]

