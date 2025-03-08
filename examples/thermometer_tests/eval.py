import numpy as np
from thermometer import *

# Erzeuge Input-Daten: von -5.0 bis unter 10.0 in Schritten von 2.5
inp = np.arange(-5.0, 10.0, 2.5)
NUM_BITS = 50

print("Input Array:")
print(inp)

# ----------------------------
# Thermometer-Encodings
# ----------------------------
thermometer = Thermometer(num_bits=NUM_BITS, feature_wise=False)
thermometer.fit(inp)

dist_thermometer = DistributiveThermometer(num_bits=NUM_BITS, feature_wise=False)
dist_thermometer.fit(inp)

gauss_thermometer = GaussianThermometer(num_bits=NUM_BITS, feature_wise=False)
gauss_thermometer.fit(inp)

out_thermometer = thermometer.binarize(inp)
out_dist_thermometer = dist_thermometer.binarize(inp)
out_gauss_thermometer = gauss_thermometer.binarize(inp)

print("\nThermometer encoding:")
print(out_thermometer)
print("\nDistributive Thermometer encoding:")
print(out_dist_thermometer)
print("\nGaussian Thermometer encoding:")
print(out_gauss_thermometer)

print("\nThresholds:")
print("Thermometer: ")
print(thermometer.thresholds)
print("\nDistributive Thermometer: ")
print(dist_thermometer.thresholds)
print("\nGaussian Thermometer: ")
print(gauss_thermometer.thresholds)

# ----------------------------
# BCD Encoding
# ----------------------------
# Hier verwenden wir num_digits=2, sodass jeder quantisierte Wert in zwei Dezimalziffern (je 4 Bit) codiert wird.
bcd_encoder = BCD(num_digits=2, feature_wise=False)
bcd_encoder.fit(inp)
out_bcd = bcd_encoder.binarize(inp)

print("\nBCD encoding:")
print(out_bcd)
print("BCD encoder range (min, max):", bcd_encoder.min_value, bcd_encoder.max_value)

# ----------------------------
# Excess-n Encoding
# ----------------------------
# Verwenden wir hier 8 Bits für die Excess-n-Codierung.
excess_encoder = ExcessN(num_bits=8, feature_wise=False)
excess_encoder.fit(inp)
out_excess = excess_encoder.binarize(inp)

print("\nExcess-N encoding:")
print(out_excess)
print("Excess-N encoder range (min, max):", excess_encoder.min_value, excess_encoder.max_value)
