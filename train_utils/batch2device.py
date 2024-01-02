def transfer_batch_to_device(batch, device, dataloader_idx):
    if "input" in batch:
        batch["input"] = batch["input"].to(device)
    if "label" in batch:
        batch["label"] = batch["label"].to(device)
    if "canny" in batch:
        batch["canny"] = batch["canny"].to(device)
    if "label_seg" in batch:
        batch["label_seg"] = batch["label_seg"].to(device)
    if "label_edge" in batch:
        batch["label_edge"] = batch["label_edge"].to(device)
    if "label_context" in batch:
        batch["label_context"] = batch["label_context"].to(device)
    return batch
