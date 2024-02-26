ds_config = tfds.rlds.rlds_base.DatasetConfig(
    name='catch_example',
    observation_info=tfds.features.Tensor(
        shape=(10, 5),
        dtype=tf.float32,
        encoding=tfds.features.Encoding.ZLIB),
    action_info=tf.int64,
    reward_info=tf.float64,
    discount_info=tf.float64,
    step_metadata_info={'timestamp_ns': tf.int64})

with envlogger.EnvLogger(
    env,
    backend = tfds_backend_writer.TFDSBackendWriter(
    data_directory=data_dir,
    split_name='train',
    max_episodes_per_file=max_episodes_per_shard,
    ds_config=ds_config),
    step_fn=step_fn) as env:
print('Done wrapping environment with EnvironmentLogger.')

print(f'Training a random agent for {num_episodes} episodes...')
for i in range(num_episodes):
    print(f'episode {i}')
    timestep = env.reset()
    while not timestep.last():
    action = np.random.randint(low=0, high=3)
    timestep = env.step(action)
print(f'Done training a random agent for {num_episodes} episodes.')
