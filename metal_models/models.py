import json
import numpy as np
import random as random
import keras


class BaseModel:
    def __init__(self, filename):
        self.START_OF_SEQ = "~"
        self.END_OF_SEQ = "[END]"
        self.PADDING = "#"

    def generate(self):
        pass

    def last(self, lst, n):
        if n == 0:
            return tuple()
        if len(lst) < n:
            return tuple([self.START_OF_SEQ] * (n - len(lst)) + lst)
        return tuple(lst[-n:])

    @staticmethod
    def sample(p, T=1):

        if T == 0:
            return np.argmax(p)

        lp = p / np.sum(p)
        lp = np.log(lp)
        lp = lp / T
        lp = np.exp(lp)
        lp = lp / np.sum(lp)
        rnd = random.random()
        accum = 0
        for n, q in enumerate(lp):
            accum += q
            if rnd <= accum:
                return n

    def pretty_print(self, tokens):
        out = ""
        for t in tokens:
            if t in ".,):!?":
                if out[-1] == " ":
                    out = out[:-1]
                out += t + " "
            elif t in "(":
                out += t
            elif t in "-\n":
                if out[-1] == " ":
                    out = out[:-1]
                out += t
            else:
                out += t + " "
        return out


class MarkovMetalMachineCharacters(BaseModel):
    def __init__(self, filename):
        super().__init__(filename)
        with open(filename, "r") as f:
            self.model = json.load(f)

        self.size = len(list(self.model.keys())[0])

    def generate(self, n=200, seed=None, T=1.0):
        if seed is None:
            sequence = [self.START_OF_SEQ] * self.size
            out = []
        else:
            sequence = list(seed)
            out = sequence[:]

        for i in range(n):
            current_state = self.last(sequence, self.size)
            possible_tokens = tuple(self.model["".join(current_state)].items())
            p = [x[1] for x in possible_tokens]
            next_token = possible_tokens[self.sample(p, T)][0]

            sequence.append(next_token)
            out.append(next_token)

            if next_token == self.END_OF_SEQ:
                return "".join(out)

        return "".join(out)


class MarkovMetalMachineWords(BaseModel):
    def __init__(self, filename):
        super().__init__(filename)
        with open(filename, "r") as f:
            self.model = json.load(f)

        self.size = len(list(self.model.keys())[0].split(" "))

    def generate(self, n=200, seed=None, T=1.0):
        if seed is None:
            sequence = [self.START_OF_SEQ] * self.size
            out = []
        else:
            sequence = list(seed.split(" "))
            out = sequence[:]

        for i in range(n):
            current_state = self.last(sequence, self.size)
            possible_tokens = tuple(self.model[" ".join(current_state)].items())
            p = [x[1] for x in possible_tokens]
            next_token = possible_tokens[self.sample(p, T)][0]

            sequence.append(next_token)
            out.append(next_token)

            if next_token == self.END_OF_SEQ:
                return self.pretty_print(out)

        return self.pretty_print(out)


class BattleMetalBrain(BaseModel):
    def __init__(self, filename):
        super().__init__(filename)

        json_model = filename + "model.json"
        model_weights = filename + "weights.h5"

        with open(json_model, "r") as f:
            self.model = keras.models.model_from_json(f.read())
            f.close()

        self.model.load_weights(model_weights)

        self.size = 10 # Hardwired size as only deploying one model.

        self.characters = sorted(
            list("abcdefghijklmnopqrstuvwxyz1234567890 \n.,():;-!?'\"")
            + [self.START_OF_SEQ, self.END_OF_SEQ])
        self.characters = [self.PADDING] + self.characters
        self.N_CHARACTERS = len(self.characters)

        self.char_to_idx = {c: n for n, c in enumerate(self.characters)}
        self.idx_to_chr = {n: c for c, n in self.char_to_idx.items()}

        self.generate()

    def string_to_array(self, s):
        return np.asarray([self.char_to_idx[c] for c in s])

    def generate(self, n=200, seed=None, T=1.0):
        if seed is None:
            sequence = [self.START_OF_SEQ] * self.size
            out = []
        else:
            sequence = list(seed)
            out = sequence[:]

        for i in range(n):
            current_state = self.last(sequence, self.size)
            X = self.string_to_array(current_state).reshape(1, -1)
            next_token = self.idx_to_chr[
                self.sample(self.model.predict_proba(X, verbose=0)[0], T=T)]

            sequence.append(next_token)
            out.append(next_token)

            if next_token == self.END_OF_SEQ:
                return "".join(out)

        return "".join(out)

class RecurrentRockRadio(BaseModel):
    def __init__(self, filename):
        super().__init__(filename)

        json_model = filename + "model.json"
        model_weights = filename + "weights.h5"

        with open(json_model, "r") as f:
            self.model = keras.models.model_from_json(f.read())
            f.close()

        self.model.load_weights(model_weights)

        self.characters = sorted(
            list("abcdefghijklmnopqrstuvwxyz1234567890 \n.,():;-!?'\"")
            + [self.START_OF_SEQ, self.END_OF_SEQ])
        self.characters = [self.PADDING] + self.characters
        self.N_CHARACTERS = len(self.characters)

        self.char_to_idx = {c: n for n, c in enumerate(self.characters)}
        self.idx_to_chr = {n: c for c, n in self.char_to_idx.items()}

        self.generate()

    def generate(self, n=200, seed=None, T=1.0):
        self.model.reset_states()

        if seed is None:
            sequence = [self.START_OF_SEQ]
            out = []
        else:
            sequence = list(seed)
            out = sequence[:]
            vec_in = np.asarray([[self.char_to_idx[self.START_OF_SEQ]]])
            self.model.predict(vec_in)
            for c in seed[:-1]:
                vec_in = np.asarray([[self.char_to_idx[c]]])
                self.model.predict(vec_in)

        for i in range(n):
            current_state = sequence[-1]
            X = np.asarray([[self.char_to_idx[current_state]]])
            next_token = self.idx_to_chr[
                self.sample(self.model.predict_proba(X, verbose=0)[0, 0], T=T)]

            sequence.append(next_token)
            out.append(next_token)

            if next_token == self.END_OF_SEQ:
                return "".join(out)

        return "".join(out)


class RecurrentRockRadioStyle(BaseModel):
    def __init__(self, filename):
        super().__init__(filename)

        json_model = filename + "model.json"
        model_weights = filename + "weights.h5"
        band_file = filename + "bands.json"

        self.bands = json.load(open(band_file, "r"))
        self.bands = sorted(self.bands)
        self.band_to_idx = {b: n for n, b in enumerate(self.bands)}
        self.idx_to_band = {n: b for n, b in enumerate(self.bands)}
        self.N_BANDS = len(self.bands)

        with open(json_model, "r") as f:
            self.model = keras.models.model_from_json(f.read())
            f.close()

        self.model.load_weights(model_weights)

        self.characters = sorted(
            list("abcdefghijklmnopqrstuvwxyz1234567890 \n.,():;-!?'\"")
            + [self.START_OF_SEQ, self.END_OF_SEQ])
        self.characters = [self.PADDING] + self.characters
        self.N_CHARACTERS = len(self.characters)

        self.char_to_idx = {c: n for n, c in enumerate(self.characters)}
        self.idx_to_chr = {n: c for c, n in self.char_to_idx.items()}

        self.generate()

    def generate(self, n=200, seed=None, T=1.0, band=None):
        self.model.reset_states()

        if band is None:
            band = np.random.randint(0, self.N_BANDS)

        if seed is None:
            sequence = [self.START_OF_SEQ]
            out = []
        else:
            sequence = list(seed)
            out = sequence[:]
            vec_in = [
                np.asarray([[self.char_to_idx[self.START_OF_SEQ]]]),
                np.asarray([[band]])
            ]
            self.model.predict(vec_in)
            for c in seed[:-1]:
                vec_in = [
                    np.asarray([[self.char_to_idx[c]]]),
                    np.asarray([[band]])
                ]
                self.model.predict(vec_in)

        for i in range(n):
            current_state = sequence[-1]
            vec_in = [
                np.asarray([[self.char_to_idx[current_state]]]),
                np.asarray([[band]])
            ]
            next_token = self.idx_to_chr[
                self.sample(self.model.predict_proba(vec_in, verbose=0)[0, 0], T=T)]

            sequence.append(next_token)
            out.append(next_token)

            if next_token == self.END_OF_SEQ:
                return "".join(out)

        return "".join(out)
