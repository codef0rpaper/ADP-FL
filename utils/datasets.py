import numpy as np
import copy

def split_df(args, data_frame, num_users):
    print("Splitting data into {} users".format(num_users))
    df_list = np.array_split(data_frame.sample(frac=1, random_state=args.seed), num_users)
    return df_list


def split_dataset(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def balance_split_dataset(all_datasets, num_total_users):
    org_dataset_size = np.array([len(dataset) for dataset in all_datasets], dtype=np.int16)
    num_items = copy.deepcopy(org_dataset_size)
    num_splits = np.array([1 for _ in range(len(all_datasets))], dtype=np.int16)

    # split the client with maximum size, until the total number is achieved
    while num_splits.sum() < num_total_users:
        split_idx = np.argmax(num_items)
        num_splits[split_idx] += 1
        num_items[split_idx] = int(org_dataset_size[split_idx] / num_splits[split_idx])

    dict_users = [{} for _ in range(len(all_datasets))]
    for idx_site in range(len(all_datasets)):
        all_idxs = np.arange(org_dataset_size[idx_site])
        np.random.shuffle(all_idxs)

        for i in range(num_splits[idx_site]):
            if i == num_splits[idx_site] - 1:
                dict_users[idx_site][i] = set(all_idxs[i * num_items[idx_site] :])
            else:
                dict_users[idx_site][i] = set(
                    all_idxs[i * num_items[idx_site] : (i + 1) * num_items[idx_site]]
                )

    return dict_users