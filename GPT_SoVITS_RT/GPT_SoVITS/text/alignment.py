import logging
import copy

class search:
    def __init__(self, flags, pos_idx, datas, task_id):
        self.flags = copy.deepcopy(flags)
        self.pos_idx = copy.deepcopy(pos_idx)
        self.datas = copy.deepcopy(datas)
        self.is_break = False
        self.task_id = task_id
    
    def add(self, del_i1, i0, i):
        data = self.datas[i0]

        for i1 in del_i1[::-1]:
            pos, idx = self.pos_idx[i1]
            data2 = self.datas[idx]
            data2['flag'] = False
            data2['pos'] = -1
            self.flags[pos:pos+data2['char_phoneme_len']] = [False]*data2['char_phoneme_len']
            self.pos_idx.pop(i1)

        self.pos_idx.append([i, i0])
        data['flag'] = True
        data['pos'] = i
        self.flags[i:i+data['char_phoneme_len']] = [True]*data['char_phoneme_len']
    
    def run1(self, i0, i, task_id):
        data = self.datas[i0]

        if True not in self.flags[i:i+data['char_phoneme_len']]:
            del_i1 = []
            for i1 in range(len(self.pos_idx)):
                pos, idx = self.pos_idx[i1]
                if i < pos:
                    del_i1.append(i1)
            
            task = None
            if len(del_i1) > 0:
                task = search(self.flags, self.pos_idx, self.datas, task_id)
                task.add(del_i1, i0, i)
                task.is_break = True
            else:
                self.add([], i0, i)

            self.is_break = True
            return task
        
        self.is_break = False
        return None
    
    def run2(self, other_phonemes, word2ph):
        for i0, data in enumerate(self.datas):
            if data['flag']:
                word2ph["word"].append(data['char'])
                pos = data['pos']
                other_ph = sum(other_phonemes[pos:pos+data['char_phoneme_len']])
                word2ph["ph"].append(data['char_phoneme_len'] + other_ph)
            else:
                i1 = i0
                while i1 >= 0:
                    if self.datas[i1]['flag']:
                        front = self.datas[i1]['pos'] + self.datas[i1]['char_phoneme_len']
                        break
                    i1 -= 1
                else:
                    front = 0
                i1 = i0
                while i1 < len(self.datas):
                    if self.datas[i1]['flag']:
                        back = self.datas[i1]['pos']
                        break
                    i1 += 1
                else:
                    back = len(other_phonemes)

                if True in self.flags[front:back]:
                    word2ph["word"][-1] += data['char']
                else:
                    self.flags[front:back] = [True]*(back-front)
                    word2ph["word"].append(data['char'])
                    other_ph = sum(other_phonemes[front:back])
                    word2ph["ph"].append(back-front + other_ph)
        
        return self.flags, word2ph

def phoneme_word2ph_alignment(phonemes, other_phonemes, char_phonemes, word2ph):
    datas = [{'flag':False, 'pos':-1, 'char':char, 'char_phoneme':char_phoneme, 'char_phoneme_len':len(char_phoneme)} for char, char_phoneme in char_phonemes]
    
    tasks = [search(
        flags=[False]*len(phonemes), 
        pos_idx=[], 
        datas=datas,
        task_id=1
    )]

    for i0, data in enumerate(datas):
        m = ' '.join(data['char_phoneme'])
        if data['char_phoneme_len'] != 0:
            for task in tasks:
                task.is_break = False
            for i in range(len(phonemes)-data['char_phoneme_len']+1):
                n = ' '.join(phonemes[i:i+data['char_phoneme_len']])
                if m == n:
                    new_tasks = []
                    for task in tasks:
                        if not task.is_break:
                            new_tasks.append(task.run1(i0, i, len(tasks)))
                    
                    if new_tasks == []:
                        break
                    
                    for task in new_tasks:
                        if not task is None:
                            tasks.append(task)
    
    flags_true_count = -1
    for task in tasks:
        _flags, _word2ph = task.run2(other_phonemes, copy.deepcopy(word2ph))
        if _flags.count(True) > flags_true_count:
            flags_true_count = _flags.count(True)
            flags = _flags
            __word2ph = _word2ph
    
    word2ph = __word2ph

    if False in flags:
        print(phonemes, char_phonemes)
        logging.warning("The phoneme word2ph cannot be aligned!")
    
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

    return word2ph
