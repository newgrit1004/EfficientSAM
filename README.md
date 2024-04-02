# reproduce step
## 1. clone this repository and set the model files.
  ```bash
    git clone --recurse-submodules https://github.com/newgrit1004/EfficientSAM.git
    cd EfficientSAM
    unzip weights/efficient_sam_vits.pt.zip
    mv efficient_sam_vits.pt ./weights
  ```

## 2. build the docker image
  ```bash
  # check the cuda version in Dockerfile. Modify the base image depending on your environment.
  docker compose -f docker-compose-build.yml build

  docker compose up -d
  docker exec -it tfjs_test /bin/bash -c "cd /workspace && /bin/bash"

  # inside the container
  python export_to_onnx.py # generated onnx files are in "./weights" folder.
  python export_onnx_to_tensorflow.py # generated tensorflow files are in "./saved_model" folder.

  # if python export_onnx_to_tensorflow.py is not executed,
  # install the onnx_tf manually inside the container.
  docker exec -it tfjs_test /bin/bash -c "cd /workspace && /bin/bash"
  root@b9fb8b01ab27:/workspace# cd onnx-tensorflow/
  root@b9fb8b01ab27:/workspace/onnx-tensorflow# pip install -e .
  root@b9fb8b01ab27:/workspace/onnx-tensorflow# cd ../
  root@b9fb8b01ab27:/workspace# python export_to_onnx.py
  root@b9fb8b01ab27:/workspace# python export_onnx_to_tensorflow.py
  ```

## 3. compare torch model and tensorflow model result
  - See the jupyter notebook "compare_tf_torch_result.ipynb" file.
  - Run the jupyter notebook in local(tensorflow 2.16.1 required)
  ```bash
  python3 -m venv .venv
  . .venv/bin/activate
  pip3 install -r requirements.txt

  # Then run the jupyter notebook cells in order.
  ```

## 4. Convert the tensorflow model into tfjs model
  ```bash
  docker compose up -d
  docker exec -it tfjs_test /bin/bash -c "cd /workspace && /bin/bash"

  # inside the container
  pip install tensorflow==2.16.1
  tensorflowjs_converter \
        --input_format tf_saved_model \
        --output_format tfjs_graph_model \
        saved_model \
        tfjs_model # generated tfjs_model files are in "./tfjs_model" folder.
  ```

## 5. See the result of tensorflow.js model inference
  Right click on index.html file then click "open with live server."


  See the result on console.
