FROM ubuntu:20.04
 
 
ARG USER=robotdataset
ARG PASSWORD=robotdataset
 
ARG CONDA_SCRIPT=Miniforge3-23.3.1-1-Linux-x86_64.sh
ARG CONDA_LINK=https://github.com/conda-forge/miniforge/releases/download/23.3.1-1/${CONDA_SCRIPT}
ARG CONDA_MD5=aef279d6baea7f67940f16aad17ebe5f6aac97487c7c03466ff01f4819e5a651

ARG pip=/home/${USER}/miniforge3/envs/${USER}/bin/pip
ARG PYTHONDONTWRITEBYTECODE=true

# Setup timezone and install base packages
RUN apt update \
&& ln -fs /usr/share/zoneinfo/Europe/Rome /etc/localtime \
&& apt install --no-install-recommends -y -qq sudo git build-essential unzip ca-certificates \
&& rm -rf /var/lib/apt/lists/*

# Create a user in the sudo group
RUN addgroup ${USER} \
&& useradd -ms /bin/bash ${USER} -g ${USER} \
&& echo "${USER}:${PASSWORD}" | chpasswd \
&& usermod -a -G sudo ${USER} \
&& sed -i.bak -e 's/%sudo\s\+ALL=(ALL\(:ALL\)\?)\s\+ALL/%sudo ALL=NOPASSWD:ALL/g' /etc/sudoers
 
USER ${USER}
WORKDIR /home/${USER}
 
# Install Mamba
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends wget bzip2 \
&& wget ${CONDA_LINK} \
&& bash ./${CONDA_SCRIPT} -b \
&& /home/${USER}/miniforge3/bin/mamba init bash \
&& sudo find /home/${USER}/miniforge3 -follow -type f -name '*.a' -delete \
&& sudo find /home/${USER}/miniforge3 -follow -type f -name '*.pyc' -delete \
&& /home/${USER}/miniforge3/bin/mamba clean -afy \
&& rm ${CONDA_SCRIPT} \
&& echo "mamba activate robotdataset" >> /home/${USER}/.bashrc

# Not sure if
RUN sudo apt update && sudo apt install pkg-config libhdf5-dev libgl1-mesa-glx libosmesa6 -y
 
COPY ./env.yml ./env.yml
RUN /home/${USER}/miniforge3/condabin/mamba env create --file env.yml && rm env.yml

# Install the oculus reader
RUN wget https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh && sudo bash script.deb.sh \
&& sudo apt install git-lfs && git lfs install \
&& git clone https://github.com/rail-berkeley/oculus_reader.git && $pip install -e ./oculus_reader \
&& sudo apt install -y android-tools-adb udev \
&& sudo usermod -aG plugdev ${USER} \
&& echo -n "SUBSYSTEM==\"usb\", ATTR{idVendor}==\"2833\", ATTR{idProduct}==\"0183\", MODE=\"0666\", GROUP=\"plugdev\"" | sudo tee /etc/udev/rules.d/51-android.rules

# Install RoboHive
RUN git clone --branch v0.6 --recursive https://github.com/vikashplus/robohive.git \
&& $pip install -e robohive[a0] && /home/${USER}/miniforge3/envs/${USER}/bin/robohive_init

# Install panda-py
RUN wget https://github.com/JeanElsner/panda-py/releases/download/v0.7.4/panda_py_0.7.4_libfranka_0.8.0.zip \
&& unzip -d ./tmp panda_py_0.7.4_libfranka_0.8.0.zip && $pip install tmp/panda_python-0.7.4+libfranka.0.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
&& rm -rf tmp panda_py_0.7.4_libfranka_0.8.0.zip

RUN sudo mv /etc/sudoers.bak /etc/sudoers
CMD ["bash"]