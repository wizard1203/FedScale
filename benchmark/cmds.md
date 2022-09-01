python docker/driver.py start benchmark/configs/femnist/conf.yml

python docker/driver.py submit benchmark/configs/femnist/conf.yml

python docker/driver.py stop femnist




python docker/driver.py start benchmark/configs/reddit/conf.yml

python docker/driver.py submit benchmark/configs/reddit/conf.yml

python docker/driver.py stop reddit



python docker/driver.py start benchmark/configs/amazon/conf.yml

python docker/driver.py submit benchmark/configs/amazon/conf.yml

python docker/driver.py stop amazon




python docker/driver.py start benchmark/configs/stackoverflow/conf.yml

python docker/driver.py submit benchmark/configs/stackoverflow/conf.yml

python docker/driver.py stop stackoverflow






cat job_name_logging |grep 'Training loss'
cat job_name_logging |grep 'FL Testing'


cat femnist_logging |grep 'Training loss'
cat femnist_logging |grep 'FL Testing'


cat reddit_logging |grep 'Training loss'
cat reddit_logging |grep 'FL Testing'


tensorboard --logdir=benchmark/logs/femnist/0812_054854 --bind_all




