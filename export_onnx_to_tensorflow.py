import onnx_tf
import onnx

onnx_model = onnx.load("./weights/efficient_sam_vits_decoder.onnx")
saved_tensorflow_folder_name = "saved_model"
tf_model = onnx_tf.backend.prepare(onnx_model, auto_cast=True)
tf_model.export_graph(saved_tensorflow_folder_name)
