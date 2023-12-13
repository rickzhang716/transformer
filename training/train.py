import time


class TrainingState:
    """Tracks the number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accumulated_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def run_epoch(
    data_iterable,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    iterations_to_accumulate=1,
    training_state=TrainingState()
):
    """Train a single epoch (a full run of the dataset). epoch = iterations * batch size"""
    start_time = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    number_accumulated = 0
    for i, batch in enumerate(data_iterable):
        output = model.encode_decode(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(output, batch.tgt_y, batch.num_tokens)

        if mode == "train" or mode == "train+log":
            loss_node.backward()
            training_state.step += 1
            training_state.samples += batch.src.shape[0]
            training_state.tokens += batch.num_tokens
            if i % iterations_to_accumulate == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                number_accumulated += 1
                training_state.accumulated_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.num_tokens
        tokens += batch.num_tokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time
            print(
                (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, number_accumulated, loss / batch.num_tokens, tokens / elapsed, lr)
            )
            start_time = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, training_state


