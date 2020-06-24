from functools import lru_cache
import ast
import operator as op

import stanza
import torch
stanza.download('en')
PIPELINE = stanza.Pipeline('en')


@lru_cache(maxsize=10000)
def nlp(text: str):
    return PIPELINE(text)


MATH_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor, ast.USub: op.neg}


def eval_expr(expr):
    return eval_(ast.parse(expr, mode='eval').body)


def eval_(node):
    if isinstance(node, ast.Num): # <number>
        return node.n
    elif isinstance(node, ast.BinOp): # <left> <operator> <right>
        return MATH_OPS[type(node.op)](eval_(node.left), eval_(node.right))
    elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
        return MATH_OPS[type(node.op)](eval_(node.operand))
    else:
        raise TypeError(node)


def sample_gpt2(model, tokenizer, cond_text, max_length=64, eos_token=' |', max_count=30, min_text_len=20):
    eos_token_id = tokenizer.encode(eos_token)[0]
    ids = [tokenizer.encode(cond_text)]
    ids = torch.tensor(ids).cuda()
    token_ids = model.generate(ids, do_sample=True, max_length=64,
                                    eos_token_id=eos_token_id)
    token_ids = token_ids[0]
    token_ids = token_ids[ids.size(1):]
    text = tokenizer.decode(tokenizer.encode('a') + token_ids.tolist())[1:]
    text = text.replace(eos_token, '').rstrip()
    if '<|endoftext|>' in text:
        idx = text.find('<|endoftext|>')
        text = text[:idx]
    if len(text) < min_text_len and max_count > 0:
        return sample_gpt2(model,
                           tokenizer,
                           cond_text,
                           max_length=max_length,
                           eos_token=eos_token,
                           max_count=max_count - 1,
                           min_text_len=min_text_len)
    else:
        return text


def sample_gpt2_mc_dialogue(model,
                            tokenizer,
                            username_target,
                            username_source,
                            source_text,
                            max_length=64,
                            max_count=30,
                            min_text_len=20):
    username_target = username_target.replace('@', '')
    cond_text = f'{username_source} {source_text} |{username_target} '
    gen_text = sample_gpt2(model,
                           tokenizer,
                           cond_text,
                           max_length=max_length,
                           max_count=max_count,
                           min_text_len=min_text_len)
    return f'<{username_target}>{gen_text}'
