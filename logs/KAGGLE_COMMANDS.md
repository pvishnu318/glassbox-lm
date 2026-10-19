!git clone https://github.com/pvishnu318/glassbox-lm.git
%cd glassbox-lm


!pip install -q -U transformers accelerate
!pip install -q -r requirements.txt


import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")


!python scripts/00_env_check.py