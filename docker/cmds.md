python docker/driver.py start benchmark/configs/femnist/conf.yml

python docker/driver.py submit benchmark/configs/femnist/conf.yml

python driver.py stop femnist


cat job_name_logging |grep 'Training loss'
cat job_name_logging |grep 'FL Testing'


cat femnist_logging |grep 'Training loss'
cat femnist_logging |grep 'FL Testing'


tensorboard --logdir=benchmark/logs/femnist/0812_054854 --bind_all




