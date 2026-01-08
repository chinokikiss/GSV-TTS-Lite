from text import symbols2 as symbols_v2


_symbol_to_id_v2 = {s: i for i, s in enumerate(symbols_v2.symbols)}


def cleaned_text_to_sequence(cleaned_text):
    phones = [_symbol_to_id_v2[symbol] for symbol in cleaned_text]
    return phones