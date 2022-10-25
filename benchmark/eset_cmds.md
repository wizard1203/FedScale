python docker/driver.py start benchmark/configs/femnist/debug.yml

python docker/driver.py submit benchmark/configs/femnist/debug.yml

python docker/driver.py stop debug



python docker/driver.py start benchmark/configs/femnist/eset_conf.yml

python docker/driver.py submit benchmark/configs/femnist/eset_conf.yml

python docker/driver.py stop femnist




python docker/driver.py start benchmark/configs/femnist/e_fem_c100_k16_fs.yml
python docker/driver.py submit benchmark/configs/femnist/e_fem_c100_k16_fs.yml
python docker/driver.py stop e_fem_c100_k16_fs


python docker/driver.py start benchmark/configs/femnist/e_fem_c100_k32_fs.yml
python docker/driver.py submit benchmark/configs/femnist/e_fem_c100_k32_fs.yml
python docker/driver.py stop e_fem_c100_k32_fs







python docker/driver.py start benchmark/configs/imagenet/e_img_c100_k4_fs.yml

python docker/driver.py submit benchmark/configs/imagenet/e_img_c100_k4_fs.yml

python docker/driver.py stop e_img_c100_k4_fs





python docker/driver.py start benchmark/configs/imagenet/e_img_c100_k8_fs.yml

python docker/driver.py submit benchmark/configs/imagenet/e_img_c100_k8_fs.yml

python docker/driver.py stop e_img_c100_k8_fs




python docker/driver.py start benchmark/configs/imagenet/e_img_c100_k16_fs.yml

python docker/driver.py submit benchmark/configs/imagenet/e_img_c100_k16_fs.yml

python docker/driver.py stop e_img_c100_k16_fs









cat job_name_logging |grep 'Training loss'
cat job_name_logging |grep 'FL Testing'


cat femnist_logging |grep 'Training loss'
cat femnist_logging |grep 'FL Testing'


cat reddit_logging |grep 'Training loss'
cat reddit_logging |grep 'FL Testing'








