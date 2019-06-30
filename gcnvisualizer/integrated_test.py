#!/usr/bin/env python
from pathlib import Path
import shutil
import subprocess as sp
from logging import (getLogger,
                     StreamHandler,
                     Formatter)


def get_logger(logname, loglevel='DEBUG'):
    _logger = getLogger(logname)
    handler = StreamHandler()
    handler.setLevel(loglevel)
    fmt = Formatter("%(asctime)s %(levelname)6s %(funcName)14s : %(message)s")
    handler.setFormatter(fmt)
    _logger.setLevel(loglevel)
    _logger.addHandler(handler)
    _logger.propagate = False
    return _logger
logger = get_logger("Integrated Test")

root_dir = Path(__file__).expanduser().resolve().parent.parent
logger.info("root direcotry %s" % str(root_dir))

##### 1. check python environemnt
logger.info("START - check python envrionment.")
try:
    import matplotlib
    logger.debug("OK - matplotlib")    
    import sklearn
    logger.debug("OK - import sklearn")        
    import tensorflow
    logger.debug("OK - import tensorflow")            
except (ImportError, ModuleNotFoundError)  as e:
    msg = "Check your python environment: {}".format(str(e))
    raise Exception(msg)
logger.info("END - checked python envrionment.")


##### 2. Download ChEMBL Dataset in multimodal dir.
logger.info("Download - ChEMBL Dataset on multimodal task.")

multimodal_dir = root_dir / 'sample_chem/multimodal/'
if not (multimodal_dir / 'CheMBL_dragon').exists():
    cmd = (f'( cd {str(multimodal_dir)} ;'
           ' sh get_dataset.sh)')
    logger.info('CMD - %s' % cmd)
    sp.run([cmd], shell=True)
else:
    logger.info('SKIP Downloading Dataset')
logger.info("END - ChEMBL Dataset on multimodal taks.")

##### 2.1 target dir
logger.info("CLEAN - unnecessary dirctories.")

original_dir = root_dir / 'sample_chem/multimodal/CheMBL_dragon/gpcr'
target_dir = '5HT1A_HUMAN'
for idir in original_dir.glob('*'):
    if target_dir == str(idir):
        logger.info('store target directory: %s' % target_dir)
        continue
    logger.info('remove: %s ' % idir)
    # shutil.rmtree(idir.resolve())
    
logger.info("CLEANED - unnecessary dirctories.")


##### 2.2 create joblib files.
logger.info("CREATE - joblib files on mutlimodal task..")

joblib_files = [('sample_sdf_profeat.jbl', 'init_sdf_profeat.sh'),
                ('sample_ecfp_profeat.jbl', 'init_ecfp_profeat.sh'),
                ('sample_sdf_sequence.jbl', 'init_sdf_sequence.sh'),
                ('sample_dragon_profeat.jbl', 'init_dragon_profeat.sh')]

for jbl, sh in joblib_files:
    if not (multimodal_dir / jbl).exists():
        cmd = (f'( cd {str(multimodal_dir)} ;'
               ' sh {sh})')
        logger.info('CMD - %s' % cmd)
        sp.run([cmd], shell=True)
    else:
        logger.info(f'SKIP - create a {jbl}.')

logger.info("COMPLATED - joblib files on mutlimodal task..")


##### 3. Download ChEMBL Dataset in multitask dir.
logger.info("Download - ChEMBL Dataset on multitask.")

multitask_dir = root_dir / 'sample_chem/multitask/'
if not (multitask_dir / 'tox21.csv').exists() or not (multitask_dir / 'tox21dragon.csv').exists():
    cmd = (f'( cd {str(multitask_dir)} ;'
           ' sh get_dataset.sh)')
    logger.info('CMD - %s' % cmd)
    sp.run([cmd,], shell=True)
else:
    logger.info('SKIP Downloading Dataset')
logger.info("END - ChEMBL Dataset on multitask.")


##### 4. Training a model on multimodal task

# simple training
cmds = ['python gcn.py --config ./example_config/sample.json train_cv',] # 

# multimodal training
cmds += ['python gcn.py --config ./sample_chem/multimodal/sample_config/multimodal_dragon_profeat.json train_cv',    # OK
         'python gcn.py --config ./sample_chem/multimodal/sample_config/multimodal_ecfp_profeat.json train_cv', # OK
         'python gcn.py --config ./sample_chem/multimodal/sample_config/multimodal_ecfp_profeat_init.json train_cv', # OK
         'python gcn.py --config ./sample_chem/multimodal/sample_config/multimodal_sdf_profeat.json train_cv', # OK
         'python gcn.py --config ./sample_chem/multimodal/sample_config/multimodal_sdf_sequence.json train_cv'] # OK, if version of cudnn is over 7.1.5. 

for cmd in cmds:
    logger.info('CMD - %s' % cmd)
    sp.run([f"(cd {root_dir}; {cmd})"], shell=True)
    


##### 5. visualization

cmds = ['python gcn.py --config ./example_config/sample.json visualize',] # 

# multimodal training
cmds += ['python gcn.py --config ./sample_chem/multimodal/sample_config/multimodal_dragon_profeat.json visualize',    # OK
         'python gcn.py --config ./sample_chem/multimodal/sample_config/multimodal_ecfp_profeat.json visualize', # OK
         'python gcn.py --config ./sample_chem/multimodal/sample_config/multimodal_ecfp_profeat_init.json visualize', # OK
         'python gcn.py --config ./sample_chem/multimodal/sample_config/multimodal_sdf_profeat.json visualize', # OK
         'python gcn.py --config ./sample_chem/multimodal/sample_config/multimodal_sdf_sequence.json visualize'] # OK, if version of cudnn is over 7.1.5. 
