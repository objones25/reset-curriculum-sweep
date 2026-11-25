import warp as wp


def pick_device() -> str:
    return "cuda:0" if wp.is_cuda_available() else "cpu"
