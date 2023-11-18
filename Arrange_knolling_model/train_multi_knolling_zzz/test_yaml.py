import yaml

running_name = 'devoted-terrain-29'
transformer_model_path = '../../ASSET/models/devoted-terrain-29'

with open(transformer_model_path + '/config-' + running_name + '.yaml', 'r') as file:
    read_data = yaml.safe_load(file)
    print('here')