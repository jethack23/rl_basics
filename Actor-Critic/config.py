class Config:
    env_name = "CartPole-v0"

    input_shape = (4,)

    q_fc_archi = [16, 32, 64]

    pi_fc_archi = [16, 32, 64]

    max_ep = 1000

    lr_q = 0.001

    lr_pi = 0.001

    df = 0.99