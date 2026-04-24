from kunserve.request import RequestOutput


def calculate_metrics(request_output: RequestOutput, request_start_time: float):
    outputs = request_output.outputs
    queue_time = request_output.start_time - request_start_time
    ttft = outputs[0].new_token_time - request_start_time
    decode_time = outputs[-1].new_token_time - outputs[0].new_token_time
    e2e_time = outputs[-1].new_token_time - request_start_time
    token_time = [output.new_token_time for output in outputs]
    if len(outputs) > 1:
        tbt = decode_time / (len(outputs) - 1)
    else:
        tbt = 0
    return {
        "request_id": request_output.request_id,
        "arrival_time": request_start_time,
        "input_len": len(request_output.prompt_token_ids),
        "output_len": len(outputs),
        "queue": queue_time,
        "ttft": ttft,
        "tbt": tbt,
        "e2e": e2e_time,
        "token_time": token_time,
    }