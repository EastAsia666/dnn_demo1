# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM tensorflow/tensorflow:1.5.0

RUN mkdir /data/dnn_samples
RUN mkdir /data/dnn_samples/train
RUN mkdir /data/dnn_samples/valid
RUN mkdir /data/dnn_samples/test
RUN mkdir /data/checkpoint_dir
RUN mkdir /data/result_dir
RUN mkdir /data/log_dir
ADD . /var/dnn_demo
ENTRYPOINT ["python", "/var/dnn_demo/main.py"]
