import torch

class Thermometer:
    def __init__(self, num_bits=1, feature_wise=True):
        assert num_bits > 0
        assert type(feature_wise) is bool

        self.num_bits = int(num_bits)
        self.feature_wise = feature_wise
        self.thresholds = None

    def get_thresholds(self, x):
        min_value = x.min(dim=0)[0] if self.feature_wise else x.min()
        max_value = x.max(dim=0)[0] if self.feature_wise else x.max()
        return min_value.unsqueeze(-1) + torch.arange(1, self.num_bits+1).unsqueeze(0) * (
            (max_value - min_value) / (self.num_bits + 1)
        ).unsqueeze(-1)

    def fit(self, x):
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
        self.thresholds = self.get_thresholds(x)
        return self

    def binarize(self, x):
        if self.thresholds is None:
            raise Exception('need to fit before calling apply')
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
        x = x.unsqueeze(-1)
        return (x > self.thresholds).float()

class GaussianThermometer(Thermometer):
    def __init__(self, num_bits=1, feature_wise=True):
        super().__init__(num_bits, feature_wise)

    def get_thresholds(self, x):
        std_skews = torch.distributions.Normal(0, 1).icdf(torch.arange(1, self.num_bits+1) / (self.num_bits+1))
        mean = x.mean(dim=0) if self.feature_wise else x.mean()
        std = x.std(dim=0) if self.feature_wise else x.std()
        thresholds = torch.stack([std_skew * std + mean for std_skew in std_skews], dim=-1)
        return thresholds

class DistributiveThermometer(Thermometer):
    def __init__(self, num_bits=1, feature_wise=True):
        super().__init__(num_bits, feature_wise)

    def get_thresholds(self, x):
        data = torch.sort(x.flatten())[0] if not self.feature_wise else torch.sort(x, dim=0)[0]
        indicies = torch.tensor([int(data.shape[0] * i / (self.num_bits+1)) for i in range(1, self.num_bits+1)])
        thresholds = data[indicies]
        return torch.permute(thresholds, (*list(range(1, thresholds.ndim)), 0))

# ----------------------------
# BCD (Binary Coded Decimal) Encoder
# ----------------------------
# Hier wird zunächst der Wertebereich mittels fit bestimmt.
# Anschließend wird jeder (normierte) Wert in einen ganzzahligen Wert im Bereich [0, 10**num_digits-1] quantisiert.
# Jede Ziffer wird dann in ihre 4-Bit-Binärdarstellung (Standard-BCD) überführt.
class BCD:
    def __init__(self, num_digits=2, feature_wise=True):
        self.num_digits = int(num_digits)
        self.feature_wise = feature_wise
        self.min_value = None
        self.max_value = None

    def fit(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        if self.feature_wise:
            self.min_value = x.min(dim=0)[0]
            self.max_value = x.max(dim=0)[0]
        else:
            self.min_value = x.min()
            self.max_value = x.max()
        return self

    def binarize(self, x):
        if self.min_value is None or self.max_value is None:
            raise Exception("Need to fit before calling binarize")
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        # Normiere x auf den Bereich [0, 1]
        norm = (x - self.min_value) / (self.max_value - self.min_value)
        max_int = 10 ** self.num_digits - 1
        quantized = torch.round(norm * max_int).long()
        # Für jede Ziffer: Zerlegung in einzelne Dezimalziffern und Umwandlung in 4-Bit
        digits = []
        for d in range(self.num_digits):
            divisor = 10 ** (self.num_digits - d - 1)
            digit = (quantized // divisor) % 10
            # 4-Bit-Binärdarstellung: Bitgewichte [8, 4, 2, 1]
            weights = torch.tensor([8, 4, 2, 1], device=digit.device, dtype=torch.long)
            digit_bits = ((digit.unsqueeze(-1) // weights) % 2).float()
            digits.append(digit_bits)
        # Zusammenfügen der einzelnen Ziffern: Ergebnis hat am Ende shape (..., num_digits*4)
        bcd_encoding = torch.cat(digits, dim=-1)
        return bcd_encoding

# ----------------------------
# Excess-n Encoder
# ----------------------------
# Hier wird der Wertebereich (inkl. möglicher negativer Werte) zunächst mittels fit ermittelt.
# Anschließend wird x in einen quantisierten ganzzahligen Bereich [-max_int, max_int] gemappt.
# Durch Addition eines Bias (hier: bias = 2^(num_bits-1) - 1) wird der Bereich in [0, 2**num_bits-1] verschoben.
# Zum Schluss erfolgt die binäre Darstellung über eine vektorisierte Bit-Aufspaltung.
class ExcessN:
    def __init__(self, num_bits=8, feature_wise=True):
        self.num_bits = int(num_bits)
        self.feature_wise = feature_wise
        self.min_value = None
        self.max_value = None
        # Bias gemäß Standard: bias = 2^(num_bits-1) - 1
        self.bias = 2 ** (self.num_bits - 1) - 1

    def fit(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        if self.feature_wise:
            self.min_value = x.min(dim=0)[0]
            self.max_value = x.max(dim=0)[0]
        else:
            self.min_value = x.min()
            self.max_value = x.max()
        return self

    def binarize(self, x):
        if self.min_value is None or self.max_value is None:
            raise Exception("Need to fit before calling binarize")
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        # Für eine symmetrische Quantisierung: Bestimme den maximalen Absolutwert
        max_abs = torch.max(torch.abs(self.min_value), torch.abs(self.max_value))
        # Maximale positive Zahl, die mit num_bits (einen Vorzeichenbit implizit abgesetzt) darstellbar ist
        max_int = 2 ** (self.num_bits - 1) - 1
        # Quantisierung: x wird in den ganzzahligen Bereich [-max_int, max_int] gemappt
        quantized = torch.round(x / max_abs * max_int).long()
        # Excess-n-Codierung: Addition des Bias
        stored = quantized + self.bias
        # Binäre Darstellung: Umwandlung in ein Bit-Array mit Länge num_bits
        weights = 2 ** torch.arange(self.num_bits - 1, -1, -1, device=stored.device, dtype=stored.dtype)
        excess_encoding = ((stored.unsqueeze(-1) // weights) % 2).float()
        return excess_encoding
