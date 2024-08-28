import numpy as np
from utils.misc import turn_on_dropout, turn_off_dropout
import constants
import torch


def least_confidence(output):  # np array
    max_prob = np.max(output, axis=-1)
    return 1 - max_prob  


def margin_sampling(output):  # top2 class margin
    margin = np.diff(-np.sort(output)[:, :, ::-1][:, :, :2]).squeeze()  # diff
    return 1 - margin  


def entropy(output):  # H,W,C
    entro = -np.nansum(np.multiply(output, np.log(output + 1e-12)), axis=-1)
    return entro 


def dropout(model, input, device, num_classes):
    model.apply(turn_on_dropout)

    B = input.shape[0]  # batch size
    entropy_maps = torch.FloatTensor(B, input.shape[2], input.shape[3]).fill_(0).to(device)

    outputs = torch.FloatTensor(B, constants.MC_STEPS, input.shape[2], input.shape[3]).to(device)  # (20,H,W)
    for step in range(constants.MC_STEPS): 
        outputs[:, step, :, :] = torch.argmax(model(input)[0], dim=1)  # B,h,w

    model.apply(turn_off_dropout)

    for i in range(B):  # outputs: B,20,H,W
        # entropy
        for c in range(num_classes):  # C
            # MC_STEPS 
            p = torch.sum(outputs[i] == c, dim=0, dtype=torch.float32) / constants.MC_STEPS  # H,W
            entropy_maps[i] = entropy_maps[i] - (p * torch.log2(p + 1e-12)) 

    entropy_maps = entropy_maps.detach().cpu().numpy()

    return entropy_maps[0]  


def topk_sample(k):
    arr = np.array([1, 30, 2, 40, 50])
    topk_idxs = arr.argsort()[-k:][::-1]  
    topk_vals = arr[topk_idxs]
    print(topk_idxs)
    print(topk_vals)


from utils.misc import minmax_normalize

if __name__ == '__main__':
    fake_probs = np.array([
        [0.1, 0.4, 0.5],
        [0.2, 0.4, 0.4],
        [0.8, 0.1, 0.1]
    ])
    a = np.array([[0, 2],  # 2x2,
                  [1, 0]]).astype(int)
    output = fake_probs[a]

    # max_prob = least_confidence(output)
    # print(max_prob)

    margin = margin_sampling(output)
    print(margin)

    # entro = entropy(output)
    # print(entro)
