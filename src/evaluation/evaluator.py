import time
import math
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.model.knn import Knn
from src.learn.database import OcrCollateFn, OcrDataset
from src.common.tokenizer import Tokenizer

from .evalconfig import EvalConfig
from .metrics import OCRMetrics


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                     Settings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load JSON config file
config_path = Path("evalconfig.json")
config = EvalConfig.model_validate_json(config_path.read_text())

# Setup the tokenizer
tokenizer = Tokenizer(Path(config.data.path_to_vocabulary_file))

# Get ids for special characters used in decoder
PAD_IDX = tokenizer.encode_special_token("<pad>")
BOS_IDX = tokenizer.encode_special_token("<bos>")
EOS_IDX = tokenizer.encode_special_token("<eos>")

# Device on which the model will run
DEVICE = config.model.device

# Load all the paths
PATH_TO_DB = config.data.path_to_db
PATH_TO_META_DB = config.data.path_to_tst_meta_db
PATH_TO_VOCABULARY = config.data.path_to_vocabulary_file
PATH_TO_MODEL = config.model.path_to_model
PATH_TO_OUTPUT_FILE = config.output.path_to_output_file

# Setup parameters for the model itself (recreate the model)
TARGET_HEIGHT = config.data.image_target_height
BATCH_SIZE = config.data.batch_size
VOCAB_SIZE = tokenizer.get_vocab_size()
IS_PRETRAIN_SWIN = config.model.is_pretrain_swin
MAX_NEW_TOKENS = config.model.max_seq_len
PAD_COLOR = config.model.img_pad_value


def is_device_cuda(device: str) -> bool:
    """
    Helper function to check, if the device is set to 'cuda'.

    :param device: Name of the device
    :returns: True if the device is 'cuda'
    """
    return device == "cuda"


def print_progress_info(ocr_metrics: OCRMetrics, current_batch_num: int, batch_total: int) -> None:
    """
    Print progress information while evaluating. The progress
    will rewrite itself, so it does not spam the console.

    :param ocr_metrics: Instance of the class OCRMetrics to count CER and WER
    :param current_batch_num: What is the current batch number
    :param batch_total: Total number of batches
    """
    error_in_text = ocr_metrics.compute()
    print(f"[{current_batch_num}/{batch_total}] Current values - CER: {error_in_text['cer']}, WER: {error_in_text['wer']}",
          end="\r", flush=True)
    
def print_final_statistics(ocr_metrics: OCRMetrics, time_start: float, time_end: float, mem_bytes: int) -> None:
    """
    Prints all the collected statistics while evaluating.
    The most important are CER, WER and used memory. Elapsed
    time can be a bit imprecise, since there are writes into files.

    :param ocr_metrics: Instance of the class OCRMetrics to count CER and WER
    :param time_start: Time when the evaluation begin
    :param time_end: Time when the evaluation stopped
    :param mem_bytes: Max allocated memory in bytes
    """
    # Compute OCR metrics CER and WER
    error_in_text = ocr_metrics.compute()
    cer = error_in_text['cer']
    wer = error_in_text['wer']

    # Compute time related metrics
    elapsed_time = time_end - time_start
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    # Messages, that will be printed out
    evaluation_time_message = f"Evaluation ended in: {minutes}m {seconds:.2f}s"
    cer_message = f"CER: {cer:.4f} => {(cer * 100):.2f}%"
    wer_message = f"WER: {wer:.4f} => {(wer * 100):.2f}%"
    mem_message = ""

    if is_device_cuda(DEVICE):
        mem_mb = mem_bytes / (1024 ** 2)
        mem_gb = mem_bytes / (1024 ** 3)
        mem_message = f"Max memory allocated: {mem_gb:.2f}GB = {mem_mb:.2f}MB"

    # Get the length of the longest message, so the
    # length of the barrier around the message has
    # the same length.
    longest_msg_len = max(
        len(evaluation_time_message),
        len(cer_message),
        len(wer_message),
        len(mem_message)
    )

    # Print the metrics
    print("\n", flush=True)
    print("-" * longest_msg_len)

    print(evaluation_time_message)
    print(cer_message)
    print(wer_message)
    if is_device_cuda(DEVICE):
        print(mem_message)
    
    print("-" * longest_msg_len)



if __name__ == "__main__":
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                    Test Dataset
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dataset = OcrDataset(
        path_to_db=PATH_TO_DB,
        path_to_meta_db=PATH_TO_META_DB,
        transform=None
    )

    collate_fn = OcrCollateFn(
        target_height=TARGET_HEIGHT,
        pad_value=PAD_COLOR
    )
    test_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                  Recreate Model
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model = Knn(
        vocab_size=VOCAB_SIZE,
        pad_token_id=PAD_IDX,
        bos_token_id=BOS_IDX,
        eos_token_id=EOS_IDX,
        is_pretrain_swin=True,
    ).to(DEVICE)

    # Load weights into the model
    checkpoint = torch.load(PATH_TO_MODEL, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set model for inference
    model.eval()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                      Metrics
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    metrics = OCRMetrics()

    if torch.cuda.is_available() and is_device_cuda(DEVICE):
        torch.cuda.reset_peak_memory_stats()

    # Begin the timer for the evaluation
    start_time = time.time()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                 Evaluation Loop
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with torch.no_grad(), open(PATH_TO_OUTPUT_FILE, "w") as output_file:
        num_of_batches = math.ceil(len(test_loader.dataset) / BATCH_SIZE)

        for batch_idx, (images, labels, _) in enumerate(test_loader):
            images = images.to(DEVICE)

            generated = model.generate(
                images=images,
                max_new_tokens=MAX_NEW_TOKENS
            )

            for i in range(generated.size(0)):
                predicted_tokens = generated[i].tolist()

                predicted_text = tokenizer.decode(predicted_tokens)
                ground_truth = labels[i]

                # Update the inner state of the OCRMetrics
                metrics.update(predicted_text, ground_truth)

                # Print the information to the console and to the output file
                print_progress_info(metrics, batch_idx, num_of_batches)
                output_file.write(f"Predicted: {predicted_text} | Target: {ground_truth}\n")
            

    # End the timer for the evaluation
    end_time = time.time()

    # Print final statistics
    print_final_statistics(metrics, start_time, end_time, torch.cuda.max_memory_allocated())
