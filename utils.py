import numpy as np


def iid_partition(data, clients):
    num_items_per_client = int(len(data) / clients)
    client_dict = {}
    data_idxs = list(range(len(data)))

    for i in range(clients):
        client_dict[i] = set(np.random.choice(data_idxs, num_items_per_client, replace=False))
        data_idxs = list(set(data_idxs) - client_dict[i])

    return client_dict


def non_iid_partition(
    dataset, labels, clients, total_shards, shards_size, num_shards_per_client
):

    shard_idxs = [i for i in range(total_shards)]
    client_dict = {i: np.array([], dtype="int64") for i in range(clients)}
    idxs = np.arange(len(dataset))
    data_labels = np.array(labels)

    label_idxs = np.vstack((idxs, data_labels))
    label_idxs = label_idxs[:, label_idxs[1, :].argsort()]
    idxs = label_idxs[0, :]

    for i in range(clients):
        rand_set = set(
            np.random.choice(shard_idxs, num_shards_per_client, replace=False)
        )
        shard_idxs = list(set(shard_idxs) - rand_set)

        for rand in rand_set:
            client_dict[i] = np.concatenate(
                (client_dict[i], idxs[rand * shards_size : (rand + 1) * shards_size]),
                axis=0,
            )

    return client_dict


def get_model_ckpt(model_type):
    if model_type == "distilbert":
        return "distilbert-base-uncased"
    elif model_type == "bert":
        return "bert-base-uncased"
    elif model_type == "mobilebert":
        return "google/mobilebert-uncased"
    elif model_type == "tinybert":
        return "prajjwal1/bert-tiny"
    else:
        raise NotImplementedError
