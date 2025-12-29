## A Lightweight Scene-Adaptive Prediction Framework Based on a Spatiotemporal Large Language Model for Transportation

### Abstract

The powerful reasoning and generalization capabilities of large language models (LLMs) are crucial for next-generation intelligent transportation systems to achieve accurate traffic flow prediction. However, existing schemes that integrate LLMs with spatiotemporal traf6fic forecasting face challenges such as spatiotemporal coupling characteristics, unified representation of multi-source heterogeneous data, and the need for cross-scenario generalization. To address these challenges, this paper proposes a lightweight scene-adaptive prediction framework based on a spatiotemporal large language foundation model. Firstly, a spatiotemporal large language foundation model for transportation is constructed through structured spatiotemporal prompt engineering and full-parameter fine-tuning, achieving domain adaptation of the general-purpose LLM. Secondly, a lightweight scene-adaptive prediction framework is designed. This framework innovatively proposes a tree structure-based spatial embedding layer to capture the hierarchical spatial dependencies of traffic networks, combines it with a temporal embedding layer to learn cyclical patterns, and employs parameter-efficient fine-tuning methods such as dynamically rank-allocated Low-Rank Adaptation and bottleneck Adapters to achieve rapid adaptation to different scenarios under limited computational resources. Experiments on multiple real-world traffic datasets demonstrate that the proposed framework outperforms existing models in terms of prediction accuracy, long- and short-term prediction generalization capability, and robustness to data missingness and noise.

![](<新建 Markdown_md_files/52fab130-e490-11f0-9213-493c671b21ae.jpeg?v=1\&type=image>)

### Create ENV

`conda env create -n lighttrafficllm python=3.11`

`conda activate lighttrafficllm`

`pip install -r requirements.txt`

### Download Data

We have prepared the data, which is ready to run the code, and uploaded it to Google Drive. **Note that this is the data for the two-stage model.** For the data used to train the one-stage LLM foundation model, please see Huggingface; we have created a dataset in ShareGPT conversational format that can be used directly for fine-tuning the LLM.&#x20;

### Download Foundation Model LLM

For the one-stage foundation model, you can train it yourself using the one-stage model we uploaded. However, we recommend using the pre-trained foundation model we already have, named LightTrafficLLM-FM. LightTrafficLLM-FM is based on the Llama-3.2 series 3B model, and has been fine-tuned with all parameters using 960,000 traffic flow spatiotemporal data points. Download link: [Huggingface.](https://huggingface.co/ubinet/LightTrafficLLM-FM "LightTrafficLLM-FM")

### Execute Train and Evaluation

The command to execute the training is:

`python training_pipeline.py`



You can customize the training parameter configuration by changing the parameters in config.py.



### Model Experimental Index Results

![](<新建 Markdown_md_files/428c90d0-e48f-11f0-9213-493c671b21ae.jpeg?v=1\&type=image>)![](<新建 Markdown_md_files/428c42b0-e48f-11f0-9213-493c671b21ae.jpeg?v=1\&type=image>)



