from functools import lru_cache
import ast
import operator as op

import stanza
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
