### Background
This pipeline is based on the inference tutorial in the [Object Detection tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb) and [the training tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html). The pipelines should be able to 
1. Prepare the input for training and pack it in record format and save it in the annotations directory. Two input files are train.record and test.record. The input imaes are in the directory raw-images-simulation.
2. Fine tune the model in the directory pre-trained-model.
3. Extract the model for later inference from the checkpoint files.
4. Classify the images with help of fine tuned model. 

### Current status
The problem I am facing is that I cannot succeed to run the models from Object Detection API in Tensorflow 1.3. I succeed to run Tensorflow 2.0 to preparre data. Then I switch to Tensorflow 1.9 and train, but after that I do not suceed to extract the data from checkpoints as a graph. 

These different versions are running in different environments. I use miniconda for that.

I am using on an 8-core PC with Ubuntu 18.04. The machine has a graphic card, but it is too old to be useful. I have tried with python 3.5, as required for the Capstone project and 3.7, which is required for Tensorflow 2.0.


Step 1, above can be completed using Tensorflow 2.0. There, the tutorial is adapted to the needs and finally a script is used to convert the csv files and images to record files. The script is taken from reference [2]. The script is same as in [2] but converted with the tool 

```
tf_upgrade_v2 --infile scripts/generate_tf_record.py --outfile scripts/generate_tf_record_2.py
```
This upgrading worked well for record file generator, but not for scripts/train.py and scripts/export_inference_graph.py. Trying 
```
tf_upgrade_v2 --infile scripts/export_inference_graph.py --outfile scripts/export_inference_graph_2.py
```
to convert will result in the following error:

```
scripts/export_inference_graph.py:111:7: ERROR: Using member tf.contrib.slim in deprecated module tf.contrib. tf.contrib.slim cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
```
while 
```
tf_upgrade_v2 --infile scripts/train.py --outfile scripts/train_2.py 
```
to convert will result in the following error:

```
scripts/train.py:87:1: ERROR: Using member tf.contrib.framework.deprecated in deprecated module tf.contrib. tf.contrib.framework.deprecated cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.

```

train.py can run under tensorflow 1.8 or 1.9 with the command:

```
python scripts/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_inception_v2_coco.config
```
but the export script does not work and generates the below error message
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-802 --output_directory trained-inference-graphs/output_inference_graph_v1.pb
Traceback (most recent call last):
  File "export_inference_graph.py", line 162, in <module>
    tf.app.run()
  File "/home/said/miniconda3/envs/carnd-term1/lib/python3.5/site-packages/tensorflow/python/platform/app.py", line 125, in run
    _sys.exit(main(argv))
  File "export_inference_graph.py", line 158, in main
    write_inference_graph=FLAGS.write_inference_graph)
  File "/home/said/ML/tensorflow35/models/research/object_detection/exporter.py", line 497, in export_inference_graph
    write_inference_graph=write_inference_graph)
  File "/home/said/ML/tensorflow35/models/research/object_detection/exporter.py", line 400, in _export_inference_graph
    graph_hook_fn=graph_hook_fn)
  File "/home/said/ML/tensorflow35/models/research/object_detection/exporter.py", line 367, in build_detection_graph
    output_collection_name=output_collection_name)
  File "/home/said/ML/tensorflow35/models/research/object_detection/exporter.py", line 346, in _get_outputs_from_inputs
    output_tensors, true_image_shapes)
  File "/home/said/ML/tensorflow35/models/research/object_detection/meta_architectures/ssd_meta_arch.py", line 802, in postprocess
    prediction_dict.get('mask_predictions')
  File "/home/said/ML/tensorflow35/models/research/object_detection/meta_architectures/ssd_meta_arch.py", line 785, in _non_max_suppression_wrapper
    return tf.contrib.tpu.outside_compilation(
AttributeError: module 'tensorflow.contrib.tpu' has no attribute 'outside_compilation'
```

### To run the pipeline
To repeat the above results:

1. Tensorflow: Version 2.0/python 3.7 for generation of record files and Version 1.9/python 3.5 for training.
2. [Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb) must be installed.
3. models/research and models/research/slim must me in the PYTHONPATH
```
$ echo $PYTHONPATH
/opt/ros/melodic/lib/python2.7/dist-packages:Path-to-models/research:Path-to-models/research/slim:
```
4. ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03 must be available for initial detection of traffic lights used for labeling. On my computer, it was in ~/.keras/datasets/

5. To reduce risks of mistakes, I started with exactly same suggested model by [the training tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html). To have the model, ssd_inception_v2_coco_2018_01_28.tar.gz should be unpacked under pre-trained-model.

6. run jupyter notebook and open notebooks/training_data_preparation_tf_20.ipynb

7. set correct paths for 

8. Run the notebook. It should go through all steps and finally generate the annotated record files in the annotation directory

```
drwxr-xr-x  2 said said     4096 Dez  1 04:12 ./
drwxr-xr-x 11 said said     4096 Dez  1 02:55 ../
-rw-r--r--  1 said said      148 Nov 30 11:26 label_map.pbtxt
-rw-r--r--  1 said said    42755 Dez  1 04:23 ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.p
-rw-r--r--  1 said said    46363 Dez  1 04:24 test_annotation_list.csv
-rw-r--r--  1 said said  7927120 Dez  1 04:24 test.record
-rw-r--r--  1 said said   265607 Dez  1 04:24 train_annotation_list.csv
-rw-r--r--  1 said said 42489122 Dez  1 04:24 train.record

```

9. Open training/ssd_inception_v2_coco.config and change "/home/said/GIT/CarND-Capstone/tl_detection_pipeline" to correct path to the tl_detection_pipeline in 5 places.

10. Go to line 157 and set the desired number of steps. It is set to 20.
```
157     num_steps: 20
```

11. Start an environment with tensorflow 1.9 and Object Detection API. Make sure that the model/reserach and model/reserch/slim are in PYTHONPATH. After that, the following command should run and generate output files.


```
python scripts/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_inception_v2_coco.config

tl_detection_pipeline$ ls training/
checkpoint
events.out.tfevents.1575171682.B250M-DS3H
graph.pbtxt
model.ckpt-0.data-00000-of-00001
model.ckpt-0.index
model.ckpt-0.meta
model.ckpt-20.data-00000-of-00001
model.ckpt-20.index
model.ckpt-20.meta
pipeline.config
ssd_inception_v2_coco.config
ssd_inception_v2_coco_original.config
ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config
```
12. Now we want to export the output so that we can use the model. Again, from tl_detection_pipeline, run the following script.

```
python scripts/export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-20 --output_directory trained-inference-graphs/output_inference_graph_v1.pb
```

This should end up with the following error:

```
AttributeError: module 'tensorflow.contrib.tpu' has no attribute 'outside_compilation'
```


### References
[1] Detection tutorial https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

[2] Training tutorial https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html.