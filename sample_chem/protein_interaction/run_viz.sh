name=st0
dataset=singletask.0
kgcn-cv-splitter --config config_${name}.json --cv cv_${name}/ --use_info --without_train --without_config
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_0.jbl --model ./model_${name}/model.001.best.ckpt --cpu --visualization_header fold0 --cpu &
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_1.jbl --model ./model_${name}/model.002.best.ckpt --cpu --visualization_header fold1 --cpu & 
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_2.jbl --model ./model_${name}/model.003.best.ckpt --cpu --visualization_header fold2 --cpu &
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_3.jbl --model ./model_${name}/model.004.best.ckpt --cpu --visualization_header fold3 --cpu &
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_4.jbl --model ./model_${name}/model.005.best.ckpt --cpu --visualization_header fold4 --cpu &

#name=mm
#dataset=multimodal
#kgcn-cv-splitter --config config_${name}.json --cv cv_${name}/ --use_info --without_train --without_config
#kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_0.jbl --model ./model_${name}/model.001.best.ckpt --cpu --visualization_header fold0 --cpu &
#kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_1.jbl --model ./model_${name}/model.002.best.ckpt --cpu --visualization_header fold1 --cpu & 
#kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_2.jbl --model ./model_${name}/model.003.best.ckpt --cpu --visualization_header fold2 --cpu &
#kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_3.jbl --model ./model_${name}/model.004.best.ckpt --cpu --visualization_header fold3 --cpu &
#kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_4.jbl --model ./model_${name}/model.005.best.ckpt --cpu --visualization_header fold4 --cpu &

name=mt
dataset=multitask
kgcn-cv-splitter --config config_${name}.json --cv cv_${name}/ --use_info --without_train --without_config
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_0.jbl --model ./model_${name}/model.001.best.ckpt --cpu --visualization_header fold0 --cpu &
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_1.jbl --model ./model_${name}/model.002.best.ckpt --cpu --visualization_header fold1 --cpu & 
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_2.jbl --model ./model_${name}/model.003.best.ckpt --cpu --visualization_header fold2 --cpu &
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_3.jbl --model ./model_${name}/model.004.best.ckpt --cpu --visualization_header fold3 --cpu &
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_4.jbl --model ./model_${name}/model.005.best.ckpt --cpu --visualization_header fold4 --cpu &


wait

#kgcn visualize --config ./config_st0.json --dataset ./cv_st0/singletask.0.test_0.jbl --model ./model_st0/model.001.best.ckpt --cpu --visualization_header fold0 --ig_modal_target features 
name=st1
dataset=singletask.1
kgcn-cv-splitter --config config_${name}.json --cv cv_${name}/ --use_info --without_train --without_config
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_0.jbl --model ./model_${name}/model.001.best.ckpt --cpu --visualization_header fold0 --cpu &
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_1.jbl --model ./model_${name}/model.002.best.ckpt --cpu --visualization_header fold1 --cpu & 
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_2.jbl --model ./model_${name}/model.003.best.ckpt --cpu --visualization_header fold2 --cpu &
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_3.jbl --model ./model_${name}/model.004.best.ckpt --cpu --visualization_header fold3 --cpu &
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_4.jbl --model ./model_${name}/model.005.best.ckpt --cpu --visualization_header fold4 --cpu &

name=st2
dataset=singletask.2
kgcn-cv-splitter --config config_${name}.json --cv cv_${name}/ --use_info --without_train --without_config
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_0.jbl --model ./model_${name}/model.001.best.ckpt --cpu --visualization_header fold0 --cpu &
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_1.jbl --model ./model_${name}/model.002.best.ckpt --cpu --visualization_header fold1 --cpu & 
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_2.jbl --model ./model_${name}/model.003.best.ckpt --cpu --visualization_header fold2 --cpu &
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_3.jbl --model ./model_${name}/model.004.best.ckpt --cpu --visualization_header fold3 --cpu &
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_4.jbl --model ./model_${name}/model.005.best.ckpt --cpu --visualization_header fold4 --cpu &

name=st3
dataset=singletask.3
kgcn-cv-splitter --config config_${name}.json --cv cv_${name}/ --use_info --without_train --without_config
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_0.jbl --model ./model_${name}/model.001.best.ckpt --cpu --visualization_header fold0 --cpu &
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_1.jbl --model ./model_${name}/model.002.best.ckpt --cpu --visualization_header fold1 --cpu & 
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_2.jbl --model ./model_${name}/model.003.best.ckpt --cpu --visualization_header fold2 --cpu &
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_3.jbl --model ./model_${name}/model.004.best.ckpt --cpu --visualization_header fold3 --cpu &
kgcn visualize --config ./config_${name}.json --dataset ./cv_${name}/${dataset}.test_4.jbl --model ./model_${name}/model.005.best.ckpt --cpu --visualization_header fold4 --cpu &

wait
