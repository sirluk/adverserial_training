import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset


class TextDS(Dataset):
    def __init__(self, text, tokenizer, max_len):
        self.text = text
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, idx: int):
        tokenized_sample = self.tokenizer(self.text[idx], padding='max_length', max_length=self.max_len, truncation=True, return_tensors="pt")
        token_ids_sample = tokenized_sample['input_ids']
        token_type_ids_sample = tokenized_sample['token_type_ids']
        attention_masks_sample = tokenized_sample['attention_mask']
        return token_ids_sample, token_type_ids_sample, attention_masks_sample


def multiprocess_tokenization(text_list, tokenizer, max_len, num_workers=16):
    ds = TextDS(text_list, tokenizer, max_len)
    _loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=num_workers, drop_last=False)
    token_ids = []
    token_type_ids = []
    attention_masks = []
    for tokenized_batch, token_type_ids_batch, attention_masks_batch in _loader:
        token_ids.append(tokenized_batch)
        token_type_ids.append(token_type_ids_batch)
        attention_masks.append(attention_masks_batch)

    token_ids = torch.cat(token_ids, dim=0).squeeze(1)
    token_type_ids = torch.cat(token_type_ids, dim=0).squeeze(1)
    attention_masks = torch.cat(attention_masks, dim=0).squeeze(1)

    return token_ids, token_type_ids, attention_masks


def read_label_file(filepath):
    with open(filepath) as f:
        data = f.read()
        return {v:k for k,v in enumerate([l for l in data.split("\n") if len(l)>0])}


def _get_data_loader(
    task_key,
    protected_key,
    text_key,
    tokenizer,
    data_path,
    labels_task_path,
    labels_prot_path=None,
    batch_size=16,
    max_length=200,
    shuffle=True,
    debug=False
):

    def batch_fn(batch):
        input_ids, token_type_ids, attention_masks, labels_task = [torch.stack(l) for l in zip(*batch)]
        x = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_masks
        }
        return x, labels_task


    def batch_fn_prot(batch):
        input_ids, token_type_ids, attention_masks, labels_task, labels_prot = [torch.stack(l) for l in zip(*batch)]
        x = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_masks
        }
        return x, labels_task, labels_prot

    with open(data_path, 'rb') as file:
        data_dicts = pickle.load(file)

    if debug:
        cutoff = min(int(batch_size*10), len(data_dicts))
        data_dicts = data_dicts[:cutoff]

    keys = [task_key, protected_key, text_key]
    x = [[d[k] for k in keys] for d in data_dicts]

    data = dict(zip(keys, zip(*x)))

    input_ids, token_type_ids, attention_masks = multiprocess_tokenization(list(data[text_key]), tokenizer, max_length)

    labels_task = read_label_file(labels_task_path)
    labels_task = torch.tensor([labels_task[t] for t in data[task_key]], dtype=torch.long)

    tds = [
        input_ids,
        token_type_ids,
        attention_masks,
        labels_task
    ]

    if labels_prot_path:
        labels_prot = read_label_file(labels_prot_path)
        tds.append(torch.tensor([labels_prot[t] for t in data[protected_key]], dtype=torch.long))
        collate_fn = batch_fn_prot
    else:
        collate_fn = batch_fn

    _dataset = TensorDataset(*tds)

    _loader = DataLoader(_dataset, shuffle=shuffle, batch_size=batch_size, drop_last=False, collate_fn=collate_fn)

    return _loader


def get_data_loader_bios(
    tokenizer,
    data_path,
    labels_task_path,
    labels_prot_path=None,
    batch_size=16,
    max_length=200,
    shuffle=True,
    debug=False
):
    return _get_data_loader(
        "title",
        "gender",
        "bio",
        tokenizer,
        data_path,
        labels_task_path,
        labels_prot_path,
        batch_size,
        max_length,
        shuffle,
        debug
    )