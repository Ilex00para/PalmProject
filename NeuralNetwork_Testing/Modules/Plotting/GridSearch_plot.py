

params = [{'lr': 1e-06, 'optimizer__weight_decay': 0}, {'lr': 1e-06, 'optimizer__weight_decay': 0.01}, {'lr': 1e-06, 'optimizer__weight_decay': 0.001}, {'lr': 1e-06, 'optimizer__weight_decay': 0.0001}, {'lr': 1e-07, 'optimizer__weight_decay': 0}, {'lr': 1e-07, 'optimizer__weight_decay': 0.01}, {'lr': 1e-07, 'optimizer__weight_decay': 0.001}, {'lr': 1e-07, 'optimizer__weight_decay': 0.0001}]


def GridSearch_plot(param_list):
    n_paras = (len(param_list[0].keys()))
    print(n_paras)


if __name__ == '__main__':
    GridSearch_plot(params)