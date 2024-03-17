---
title: Building an Anime Recommendation System with PySpark in SageMaker
published: true
description: Demonstrate an anime recommendation system using PySpark within a SageMaker notebook instance.
tags: 'pyspark, sagemarker, aws, demo'
cover_image: ./assets/header.png
canonical_url: null
id: 1792862
date: '2024-03-17T10:35:27Z'
---

#### Demonstration of an Anime Recommendation System with PySpark in SageMaker `v1.0.0-alpha01`

### Understanding the Dataset
Our journey begins with understanding the dataset. We will be using the [MyAnimeList dataset](https://www.kaggle.com/datasets/azathoth42/myanimelist?datasetId=28524&sortBy=voteCount) sourced from Kaggle, which contains valuable information about anime titles, user ratings, and more. This dataset will serve as the foundation for our recommendation system.

### Preprocessing the Data
Before diving into model building, we will preprocess the dataset to ensure it is clean and structured for analysis. While the preprocessing steps have already been completed, we will briefly discuss the importance of data preprocessing in the context of recommendation systems.

for detailed preprocessing check out this notebook
[anime-recommendation-system.ipynb](https://github.com/muratsahilli/pyspark-recommendation-system/blob/main/anime-recommendation-system.ipynb) 

`TODO: Add all preprocessing step-by-step with explanations`

---

### Visualization
Visualizing the preprocessed data can provide valuable insights into the distribution of anime ratings, user preferences, and other patterns. We will utilize various visualization techniques to gain a better understanding of our dataset.

You can find data visualization techniques applied to better understand the content of the data in [anime-recommendation-system.ipynb](https://github.com/muratsahilli/pyspark-recommendation-system/blob/main/anime-recommendation-system.ipynb). You can find one of them below as example.

![image](https://github.com/muratsahilli/pyspark-recommendation-system/assets/61403011/93ba9a0f-84a4-4b48-b773-e378bec963c6)

`TODO: Add all visualizations step-by-step with explanations and insights`

---

### Recommendation System
We employed the Alternating Least Squares (ALS) algorithm to build the recommendation system using PySpark. We used the mean square error algorithm to find the best model. We calculated how accurately it measured the real value by training the data allocated for the test and estimating the remaining data.

Since these parameters had the lowest mean square error, we created a model with these parameters to obtain more accurate results.

```python
rank, iter, lambda_ = 50, 10, 0.1
model = ALS.train(rating, rank=rank, iterations=iter, lambda_=lambda_, seed=5047)
```

### Cloud Infrastructure Setup with Terraform

First of all we need to upload preproocessed data to a s3 bucket. Use the following commands. You can reach to [upload.py](https://github.com/muratsahilli/pyspark-recommendation-system/blob/main/sagemaker/upload_data.py) in there.

```sh
$ cd sagemaker
$ python upload_data.py -n 'anime-recommendation-system' -r 'eu-central-1' -f '../preprocessed_data'

../preprocessed_data\user.csv  4579798 / 4579798.0  (100.00%)00%)
```

After that run terraform commands to create an Amazon SageMaker Notebook Instance:

[instance.tf](https://github.com/muratsahilli/pyspark-recommendation-system/blob/main/sagemaker/instance.tf)

```sh
$ terraform init
Initializing the backend...

Initializing provider plugins...
- Finding latest version of hashicorp/aws...
- Installing hashicorp/aws v5.41.0...
- Installed hashicorp/aws v5.41.0 (signed by HashiCorp)
...
$ terraform plan
...
Plan: 4 to add, 0 to change, 0 to destroy.
$ terraform apply --auto-approve
aws_iam_policy.sagemaker_s3_full_access: Creating...
aws_iam_role.sagemaker_role: Creating...
aws_iam_policy.sagemaker_s3_full_access: Creation complete after 1s [id=arn:aws:iam::749270828329:policy/SageMaker_S3FullAccessPoliciy]
aws_iam_role.sagemaker_role: Creation complete after 1s [id=AnimeRecommendation_SageMakerRole]
aws_iam_role_policy_attachment.sagemaker_s3_policy_attachment: Creating...
aws_sagemaker_notebook_instance.notebookinstance: Creating...
aws_iam_role_policy_attachment.sagemaker_s3_policy_attachment: Creation complete after 1s [id=AnimeRecommendation_SageMakerRole-20240317120654686300000001]
aws_sagemaker_notebook_instance.notebookinstance: Still creating... [10s elapsed]
...
Apply complete! Resources: 4 added, 0 changed, 0 destroyed.
```

### Create a new conda_python3 notebook and Run [`sagemaker-anime-recommendation-system.ipynb`](https://github.com/muratsahilli/pyspark-recommendation-system/blob/main/sagemaker/sagemaker-anime-recommendation-system.ipynb) step by step on the Notebook Instance

You can look at the `sagemaker-anime-recommendation-system-test.html` file to see the output.

In this code, model data is saved locally and then uploaded properly to an s3 bucket.

```python
model.save(SparkContext.getOrCreate(), 'model')

import boto3
import os

# Initialize S3 client
s3 = boto3.client('s3')

# Upload files to the created bucket
bucketname = 'anime-recommendation-system'
local_directory = './model'
destination = 'model/'
for root, dirs, files in os.walk(local_directory):
    for filename in files:
        # construct the full local path
        local_path = os.path.join(root, filename)

        relative_path = os.path.relpath(local_path, local_directory)
        s3_path = os.path.join(destination, relative_path)
        
        s3.upload_file(local_path, bucketname, s3_path)
```
![image](https://github.com/akinbezatoglu/pyspark-recommendation-system/assets/61403011/05b111f2-f12b-408a-b1c4-adbfd377a07b)

![image](https://github.com/akinbezatoglu/pyspark-recommendation-system/assets/61403011/6a5a3dd6-2d5f-4aa1-88fb-a4469ccfc095)

### To load and test the model, run the [`test_another_instance.ipynb`](https://github.com/muratsahilli/pyspark-recommendation-system/blob/main/sagemaker/test_another_instance.ipynb) jupyter notebook

In this code, model data is retrieved from the s3 bucket and loaded using the Matrix Factorization Model

```python
import boto3
import os 

s3_resource = boto3.resource('s3')
bucket = s3_resource.Bucket('anime-recommendation-system') 
for obj in bucket.objects.filter(Prefix = 'model'):
    if not os.path.exists(os.path.dirname(obj.key)):
        os.makedirs(os.path.dirname(obj.key))
    bucket.download_file(obj.key, obj.key)

from pyspark import SparkContext
from pyspark.mllib.recommendation import MatrixFactorizationModel

m = MatrixFactorizationModel.load(SparkContext.getOrCreate(), 'model')
```

As a result, I explained how to create the model by loading the model data into each separate environment, making the model easier to use, and processing it with SageMaker's notebook instance. See you in another article...👋👋