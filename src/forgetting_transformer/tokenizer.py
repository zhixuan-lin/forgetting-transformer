import json
from typing import Callable, Dict, Union, Optional, Tuple, NamedTuple, Any, List
from transformers import GPT2Tokenizer

class JSONGPT2Tokenizer(GPT2Tokenizer):
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        (text, kwargs) = super().prepare_for_tokenization(text, is_split_into_words, **kwargs)
        text = json.dumps(text)
        text = text[1:-1]
        return (text, kwargs)

    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ):
        text = super().decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
        try:
            # Unfortunately this is what LongCrawl64 did. 
            text = json.loads(f'"{text}"')
        except json.JSONDecodeError:
            # Best effort decoding
            text = text.encode().decode("unicode_escape", "ignore")
        return text
