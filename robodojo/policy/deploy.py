def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name='reset')
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name='update_obs', obs=TASK_ENV.get_obs())
        actions = model_client.call(func_name='get_action')
        if len(actions) != 1:
            raise ValueError('History adapter returns exactly one control-tick action')
        TASK_ENV.take_action(actions[0])
