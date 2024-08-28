from active_selection.random_selection import RandomSelector
from active_selection.coreset import CoreSetSelector
from active_selection.softmax_entropy import SoftmaxEntropySelector
from active_selection.mc_dropout import MCDropoutEntropySelector

# from datasets.build_datasets import data_cfg

feature_dim = {
    'drn': 304
}


def get_active_selector(args):
    dataset = args.dataset
    img_size = (480, 640) 
    num_classes = 4 

    active_selection_mode = args.active_selection_mode

    if active_selection_mode == 'random':
        return RandomSelector(dataset, img_size)

    elif active_selection_mode == 'entropy':
        return SoftmaxEntropySelector(dataset, img_size)

    elif active_selection_mode == 'coreset':
        return CoreSetSelector(dataset, img_size, feature_dim[args.backbone])

    elif active_selection_mode == 'dropout':
        return MCDropoutEntropySelector(dataset, img_size, num_classes)
