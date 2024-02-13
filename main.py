from model import CofiPara
from models.util.slconfig import SLConfig
from trainer import trainer, test

model_config_path = './model_config.py'
args = SLConfig.fromfile(model_config_path) 
model = CofiPara(args)

# trainer(model,args)
test(model,args)      # if test