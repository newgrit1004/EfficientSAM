FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workdir

RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install software-properties-common -y && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt remove python-pip  python3-pip && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        ca-certificates \
        g++ \
        python3-distutils \
        python3.10 \
        python3.10-dev \
        python3.10-venv

RUN ln -s /usr/bin/python3.10 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/local/bin/python
RUN apt-get install curl wget sudo -y

#python3.8 이상부터 pip 설치할땐 하단과 같이 사용
#https://gist.github.com/jesterjunk/fe4a2780f06351c0fb2f5b6c0a347d86
RUN curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python get-pip.py
RUN pip install --upgrade pip

RUN pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu117

RUN pip install onnx==1.14.1 \
&& pip install onnxsim==0.4.33 \
&& pip install nvidia-pyindex \
&& pip install onnx_graphsurgeon \
&& pip install onnx2tf \
&& pip install simple_onnx_processing_tools \
&& pip install tensorflow==2.14.0 \
&& pip install protobuf==3.20.3 \
&& pip install h5py==3.7.0 \
&& pip install psutil==5.9.5 \
&& pip install onnxruntime-gpu==1.17.0 \
&& pip install ml_dtypes==0.2.0 \
&& pip install tensorflowjs \
pip install requests==2.27.1


#install onnx-tensorflow
COPY ./onnx-tensorflow /workspace/onnx-tensorflow
WORKDIR /workspace/onnx-tensorflow
RUN pip install -e .

# compatibility for runnning export_onnx_to_tensorflow.py script
RUN pip install tensorflow==2.13 tensorflow_probability==0.20

RUN wget https://github.com/PINTO0309/onnx2tf/releases/download/1.16.31/flatc.tar.gz \
  && tar -zxvf flatc.tar.gz \
  && sudo chmod +x flatc \
  && sudo mv flatc /usr/bin/
