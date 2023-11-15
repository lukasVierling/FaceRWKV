import types
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
from rwkv import RWKV_RNN
from tokenizers import Tokenizer
from torch.nn import functional as F

tokenizer = Tokenizer.from_file("20B_tokenizer.json")

def sample_logits(out, temperature=1.0, top_p=0.8):
    probs = F.softmax(out, dim=-1).numpy()
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), p=probs)
    return out

def main():
    args = types.SimpleNamespace()
    args.MODEL_NAME = 'RWKV-4-Pile-430M-20220808-8066'
    args.n_layer = 24
    args.n_embd = 1024

    context = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
    NUM_TRIALS = 3
    LENGTH_PER_TRIAL = 100
    TEMPERATURE = 1.0
    TOP_P = 0.85

    print(f'\nUsing CPU. Loading {args.MODEL_NAME} ...')
    model = RWKV_RNN(args)

    print(f'\nPreprocessing context (slow version. see v2/rwkv/model.py for fast version)')
    init_state = None
    for token in tokenizer.encode(context).ids:
        init_out, init_state = model.forward(token, init_state)

    for TRIAL in range(NUM_TRIALS):
        print(f'\n\n--[ Trial {TRIAL} ]-----------------', context, end="")
        all_tokens = []
        out_last = 0
        out, state = init_out.clone(), init_state.clone()
        for i in range(LENGTH_PER_TRIAL):
            token = sample_logits(out, TEMPERATURE, TOP_P)
            all_tokens += [token]
            tmp = tokenizer.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp: # only print when we have a valid utf-8 string
                print(tmp, end="", flush=True)
                out_last = i + 1
            out, state = model.forward(token, state)       
    print('\n')



if __name__ == "__main__":
    main()