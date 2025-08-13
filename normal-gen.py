import os
import time
import threading
import argparse

import torch
from model.vq_passgpt import VQPassGPTModel
from tokenizer import CharTokenizer

MAX_LEN = 32 


class ThreadBase(threading.Thread):
    def __init__(self, target=None, args=()):
        super().__init__()
        self.func = target
        self.args = args
        self.result = []  # mặc định rỗng

    def run(self):
        try:
            self.result = self.func(*self.args)
        except Exception:
            import traceback; traceback.print_exc()
            self.result = []

    def get_result(self):
        return self.result


def gen_sample(test_model_path, tokenizer, GEN_BATCH_SIZE, GPU_ID):
    if torch.cuda.is_available():
        torch.cuda.set_device(GPU_ID)
        device = torch.device(f"cuda:{GPU_ID}")
    else:
        device = torch.device("cpu")

    test_model_path = os.path.normpath(test_model_path)
    model = VQPassGPTModel.from_pretrained(
        test_model_path,
        use_vq=True,
        local_files_only=True
    ).to(device).eval()

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    BOS = tokenizer.bos_token_id
    EOS = tokenizer.eos_token_id
    PAD = tokenizer.pad_token_id

    cfg = model.config

    if hasattr(cfg, "n_positions") and MAX_LEN > int(cfg.n_positions):
        raise ValueError(f"MAX_LEN={MAX_LEN} > n_positions={cfg.n_positions}. Hãy train với n_positions >= {MAX_LEN}.")

    inputs = torch.full((GEN_BATCH_SIZE, 1), BOS, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_length=MAX_LEN,                
            do_sample=True,
            top_p=0.95,
            temperature=0.9,
            pad_token_id=PAD,
            eos_token_id=EOS,
        )

    def decode_ids(seq):
        seq = seq.tolist()
        if EOS in seq:
            seq = seq[:seq.index(EOS) + 1]
        seq = [t for t in seq if t != PAD]
        return tokenizer.decode(seq)

    texts = [decode_ids(s) for s in outputs]
    return list(set(texts))


def gen_parallel(vocab_file, batch_size, test_model_path, N, gen_passwords_path, num_gpus, gpu_index):
    print('Load tokenizer.')
    tokenizer = CharTokenizer(
        vocab_file=vocab_file,
        bos_token="<BOS>",
        eos_token="<EOS>",
        sep_token="<SEP>",
        unk_token="<UNK>",
        pad_token="<PAD>"
    )
    tokenizer.padding_side = "left"

    if not torch.cuda.is_available() and num_gpus > 0:
        print('WARNING: GPU not found, running CPU.')

    total_start = time.time()
    threads = {}
    total_passwords = []

    total_round = N // batch_size
    print('*' * 30)
    print('Generation begin.')
    print(f'Total generation needs {total_round} batchs.')

    i = 0
    while i < total_round or len(threads) > 0:
        if len(threads) == 0:
            for gpu_id in range(num_gpus):
                if i < total_round:
                    t = ThreadBase(
                        target=gen_sample,
                        args=(test_model_path, tokenizer, batch_size, gpu_id + gpu_index)
                    )
                    t.start()
                    threads[t] = i
                    i += 1

        temp_threads = list(threads.items())
        for t, idx in temp_threads:
            t.join()
            if not t.is_alive():
                new_passwords = t.get_result()  # luôn là list
                new_num = len(new_passwords)
                total_passwords += new_passwords
                print(f'[{idx + 1}/{total_round}] generated {new_num}.')
                threads.pop(t)

    total_passwords = set(total_passwords)

    os.makedirs(gen_passwords_path, exist_ok=True)
    out_file = os.path.join(gen_passwords_path, 'Normal-GEN.txt')
    with open(out_file, 'w', encoding='utf-8', errors='ignore') as f_gen:
        for pw in total_passwords:
            f_gen.write(pw + '\n')

    total_time = time.time() - total_start
    print(f'Generation file saved in: {out_file}')
    print('Generation done.')
    print('*' * 30)
    print(f'Use time:{total_time:.2f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='model/last-step', help="directory of VQPassGPT checkpoint")
    parser.add_argument("--vocabfile_path", type=str, default='./tokenizer/vocab.json', help="path of vocab file")
    parser.add_argument("--output_path", type=str, required=True, help="directory to save results")
    parser.add_argument("--generate_num", default=100, type=int, help="total number to generate")
    parser.add_argument("--batch_size", default=10, type=int, help="generate batch size")
    parser.add_argument("--gpu_num", default=1, type=int, help="number of gpus to use")
    parser.add_argument("--gpu_index", default=0, type=int, help="starting GPU index")
    args = parser.parse_args()

    model_path = args.model_path
    vocab_file = args.vocabfile_path
    output_path = os.path.join(args.output_path, str(args.generate_num))
    os.makedirs(output_path, exist_ok=True)

    gen_parallel(
        vocab_file=vocab_file,
        batch_size=args.batch_size,
        test_model_path=model_path,
        N=args.generate_num,
        gen_passwords_path=output_path,
        num_gpus=args.gpu_num,
        gpu_index=args.gpu_index
    )
