

def phoneme_word2ph_alignment(phonemes, other_phonemes, char_phonemes, word2ph):
    flags = [False]*len(phonemes)
    pos_idx = []
    datas = [{'flag':False, 'pos':-1, 'char':char, 'char_phoneme':char_phoneme, 'char_phoneme_len':len(char_phoneme)} for char, char_phoneme in char_phonemes]

    for i0, data in enumerate(datas):
        m = ' '.join(data['char_phoneme'])
        if data['char_phoneme_len'] != 0:
            for i in range(len(phonemes)-data['char_phoneme_len']+1):
                n = ' '.join(phonemes[i:i+data['char_phoneme_len']])
                if m == n and True not in flags[i:i+data['char_phoneme_len']]:
                    pos_idx.append([i, i0])
                    data['flag'] = True
                    data['pos'] = i
                    flags[i:i+data['char_phoneme_len']] = [True]*data['char_phoneme_len']

                    for i1 in range(len(pos_idx)-1, -1, -1):
                        pos, idx = pos_idx[i1]
                        if i < pos:
                            data2 = datas[idx]
                            data2['flag'] = False
                            data2['pos'] = -1
                            flags[pos:pos+data2['char_phoneme_len']] = [False]*data2['char_phoneme_len']
                            pos_idx.pop(i1)
                    
                    break

    for i0, data in enumerate(datas):
        if data['flag']:
            word2ph["word"].append(data['char'])
            pos = data['pos']
            other_ph = sum(other_phonemes[pos:pos+data['char_phoneme_len']])
            word2ph["ph"].append(data['char_phoneme_len'] + other_ph)
        else:
            i1 = i0
            while i1 >= 0:
                if datas[i1]['flag']:
                    front = datas[i1]['pos'] + datas[i1]['char_phoneme_len']
                    break
                i1 -= 1
            else:
                front = 0
            i1 = i0
            while i1 < len(datas):
                if datas[i1]['flag']:
                    back = datas[i1]['pos']
                    break
                i1 += 1
            else:
                back = len(phonemes)

            if True in flags[front:back]:
                word2ph["word"][-1] += data['char']
            else:
                flags[front:back] = [True]*(back-front)
                word2ph["word"].append(data['char'])
                other_ph = sum(other_phonemes[front:back])
                word2ph["ph"].append(back-front + other_ph)
    
    for i1, flag in enumerate(flags):
        if not flag:
            s = 0
            for i0, ph in enumerate(word2ph["ph"]):
                if s+ph > i1:
                    break
                else:
                    s += ph
            word2ph["word"].insert(i0, '')
            word2ph["ph"].insert(i0, 1)