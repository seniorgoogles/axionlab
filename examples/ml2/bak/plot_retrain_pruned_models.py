import matplotlib.pyplot as plt

data = {
    "45.75% acc drop allowed": {
        "ptp-acc": 39.031269271230094,
        "pat-acc": 72.59495048427468,
        "zero_param_per": 86.71608361774744
    },
    "13.25% acc drop allowed": {
        "ptp-acc": 62.62016178764465,
        "pat-acc": 74.99818623716763,
        "zero_param_per": 73.10153583617748
    },
    "14.5% acc drop allowed": {
        "ptp-acc": 62.62016178764465,
        "pat-acc": 75.04897159647405,
        "zero_param_per": 73.10153583617748
    },
    "9.25% acc drop allowed": {
        "ptp-acc": 67.56357238727465,
        "pat-acc": 75.19633982660427,
        "zero_param_per": 70.74445392491468
    },
    "21.75% acc drop allowed": {
        "ptp-acc": 59.12232016541517,
        "pat-acc": 74.96553850618493,
        "zero_param_per": 75.45328498293516
    },
    "18.5% acc drop allowed": {
        "ptp-acc": 61.80442195378532,
        "pat-acc": 74.97460732034679,
        "zero_param_per": 73.2721843003413
    },
    "40.0% acc drop allowed": {
        "ptp-acc": 37.26058330612689,
        "pat-acc": 73.72764537309101,
        "zero_param_per": 85.85750853242321
    },
    "14.0% acc drop allowed": {
        "ptp-acc": 62.62016178764465,
        "pat-acc": 75.07980556462437,
        "zero_param_per": 73.10153583617748
    },
    "36.25% acc drop allowed": {
        "ptp-acc": 43.43735263176987,
        "pat-acc": 74.4354663184242,
        "zero_param_per": 79.82081911262799
    },
    "13.75% acc drop allowed": {
        "ptp-acc": 62.62016178764465,
        "pat-acc": 74.93243733449414,
        "zero_param_per": 73.10153583617748
    },
    "5.75% acc drop allowed": {
        "ptp-acc": 69.8035694852541,
        "pat-acc": 75.29700366380092,
        "zero_param_per": 54.77282423208191
    },
    "40.5% acc drop allowed": {
        "ptp-acc": 35.915224725214934,
        "pat-acc": 73.35673087387093,
        "zero_param_per": 86.37478668941979
    },
    "50.0% acc drop allowed": {
        "ptp-acc": 25.850654768382487,
        "pat-acc": 72.52512061522835,
        "zero_param_per": 88.96651023890784
    },
    "45.0% acc drop allowed": {
        "ptp-acc": 39.031269271230094,
        "pat-acc": 73.2678564950847,
        "zero_param_per": 86.71608361774744
    },
    "32.75% acc drop allowed": {
        "ptp-acc": 47.153752675300176,
        "pat-acc": 73.9552726085537,
        "zero_param_per": 80.91403583617748
    },
    "39.25% acc drop allowed": {
        "ptp-acc": 37.26058330612689,
        "pat-acc": 73.22795371277252,
        "zero_param_per": 85.86284129692832
    },
    "26.25% acc drop allowed": {
        "ptp-acc": 56.34726303188595,
        "pat-acc": 74.79776544419052,
        "zero_param_per": 68.90465017064847
    },
    "8.5% acc drop allowed": {
        "ptp-acc": 67.56357238727465,
        "pat-acc": 75.28113323901766,
        "zero_param_per": 70.74445392491468
    },
    "49.75% acc drop allowed": {
        "ptp-acc": 25.850654768382487,
        "pat-acc": 72.51378459752603,
        "zero_param_per": 88.96651023890784
    },
    "37.25% acc drop allowed": {
        "ptp-acc": 38.43771538433634,
        "pat-acc": 74.15750716436318,
        "zero_param_per": 81.32465870307168
    },
    "12.5% acc drop allowed": {
        "ptp-acc": 63.69436282511699,
        "pat-acc": 75.08524685312149,
        "zero_param_per": 72.92555460750853
    },
    "10.25% acc drop allowed": {
        "ptp-acc": 65.2959154061015,
        "pat-acc": 75.15190263721117,
        "zero_param_per": 59.140358361774744
    },
    "47.75% acc drop allowed": {
        "ptp-acc": 29.19251278702797,
        "pat-acc": 73.3589980774114,
        "zero_param_per": 87.29202218430035
    },
    "32.0% acc drop allowed": {
        "ptp-acc": 47.153752675300176,
        "pat-acc": 74.48987920339536,
        "zero_param_per": 80.91403583617748
    },
    "29.0% acc drop allowed": {
        "ptp-acc": 47.153752675300176,
        "pat-acc": 74.05412268291798,
        "zero_param_per": 80.91403583617748
    },
    "16.75% acc drop allowed": {
        "ptp-acc": 61.92413030072188,
        "pat-acc": 75.07753836108391,
        "zero_param_per": 68.55802047781569
    },
    "42.5% acc drop allowed": {
        "ptp-acc": 39.031269271230094,
        "pat-acc": 72.75093408785867,
        "zero_param_per": 86.71608361774744
    },
    "21.5% acc drop allowed": {
        "ptp-acc": 58.030434940327204,
        "pat-acc": 74.98866398229768,
        "zero_param_per": 73.26685153583618
    },
    "20.25% acc drop allowed": {
        "ptp-acc": 58.675227627235465,
        "pat-acc": 75.06620234338158,
        "zero_param_per": 64.19048634812286
    },
    "2.75% acc drop allowed": {
        "ptp-acc": 72.74639968077774,
        "pat-acc": 75.37544890630102,
        "zero_param_per": 61.49744027303754
    },
    "24.0% acc drop allowed": {
        "ptp-acc": 52.66532448217071,
        "pat-acc": 74.1697500634817,
        "zero_param_per": 77.46373720136519
    },
    "21.0% acc drop allowed": {
        "ptp-acc": 54.623734900424424,
        "pat-acc": 75.03446149381507,
        "zero_param_per": 72.24296075085324
    },
    "34.25% acc drop allowed": {
        "ptp-acc": 43.43735263176987,
        "pat-acc": 74.40689955381434,
        "zero_param_per": 79.82081911262799
    },
    "31.25% acc drop allowed": {
        "ptp-acc": 47.153752675300176,
        "pat-acc": 74.15161243515797,
        "zero_param_per": 80.91403583617748
    },
    "16.0% acc drop allowed": {
        "ptp-acc": 59.5100119708347,
        "pat-acc": 75.12696339826604,
        "zero_param_per": 65.69432593856655
    },
    "10.0% acc drop allowed": {
        "ptp-acc": 67.27337033409512,
        "pat-acc": 75.25075271157543,
        "zero_param_per": 70.73912116040955
    },
    "30.25% acc drop allowed": {
        "ptp-acc": 47.153752675300176,
        "pat-acc": 74.098106431603,
        "zero_param_per": 80.91403583617748
    },
    "15.25% acc drop allowed": {
        "ptp-acc": 60.32439148256974,
        "pat-acc": 74.99002430442195,
        "zero_param_per": 73.34151023890784
    },
    "33.0% acc drop allowed": {
        "ptp-acc": 47.153752675300176,
        "pat-acc": 74.13120760329379,
        "zero_param_per": 80.91403583617748
    },
    "39.5% acc drop allowed": {
        "ptp-acc": 37.26058330612689,
        "pat-acc": 73.55171037835093,
        "zero_param_per": 85.86284129692832
    },
    "4.0% acc drop allowed": {
        "ptp-acc": 72.32288605941888,
        "pat-acc": 75.30879312221134,
        "zero_param_per": 64.19048634812286
    },
    "31.5% acc drop allowed": {
        "ptp-acc": 47.153752675300176,
        "pat-acc": 74.30668915732579,
        "zero_param_per": 80.91403583617748
    },
    "49.25% acc drop allowed": {
        "ptp-acc": 28.24391482569739,
        "pat-acc": 72.33104799216454,
        "zero_param_per": 87.8039675767918
    },
    "6.75% acc drop allowed": {
        "ptp-acc": 68.76519026372111,
        "pat-acc": 75.23805637174883,
        "zero_param_per": 67.8754266211604
    },
    "36.75% acc drop allowed": {
        "ptp-acc": 38.80908332426452,
        "pat-acc": 73.92126455544673,
        "zero_param_per": 84.18835324232083
    },
    "23.5% acc drop allowed": {
        "ptp-acc": 52.66532448217071,
        "pat-acc": 74.47355533790402,
        "zero_param_per": 77.46373720136519
    },
    "4.5% acc drop allowed": {
        "ptp-acc": 70.997478869663,
        "pat-acc": 75.25574055936445,
        "zero_param_per": 65.69432593856655
    },
    "17.5% acc drop allowed": {
        "ptp-acc": 60.039177277179235,
        "pat-acc": 75.11472049914754,
        "zero_param_per": 71.08575085324233
    },
    "37.0% acc drop allowed": {
        "ptp-acc": 38.80908332426452,
        "pat-acc": 73.87365328109696,
        "zero_param_per": 84.18835324232083
    },
    "12.25% acc drop allowed": {
        "ptp-acc": 63.69436282511699,
        "pat-acc": 75.06620234338158,
        "zero_param_per": 72.93088737201366
    },
    "33.75% acc drop allowed": {
        "ptp-acc": 43.43735263176987,
        "pat-acc": 74.32437334494142,
        "zero_param_per": 79.82081911262799
    },
    "2.0% acc drop allowed": {
        "ptp-acc": 73.6963579642326,
        "pat-acc": 75.43258243552073,
        "zero_param_per": 57.636518771331055
    },
    "25.75% acc drop allowed": {
        "ptp-acc": 56.34726303188595,
        "pat-acc": 74.70662386186382,
        "zero_param_per": 68.90465017064847
    },
    "42.25% acc drop allowed": {
        "ptp-acc": 39.031269271230094,
        "pat-acc": 73.03252076758443,
        "zero_param_per": 86.71608361774744
    },
    "49.0% acc drop allowed": {
        "ptp-acc": 29.19251278702797,
        "pat-acc": 73.29324917473791,
        "zero_param_per": 87.29202218430035
    },
    "47.0% acc drop allowed": {
        "ptp-acc": 29.571589218993726,
        "pat-acc": 73.42202633583632,
        "zero_param_per": 86.9507252559727
    },
    "37.5% acc drop allowed": {
        "ptp-acc": 38.43771538433634,
        "pat-acc": 74.20239779446439,
        "zero_param_per": 81.32465870307168
    },
    "25.0% acc drop allowed": {
        "ptp-acc": 50.676986977182864,
        "pat-acc": 74.78098813799107,
        "zero_param_per": 65.69432593856655
    },
    "30.5% acc drop allowed": {
        "ptp-acc": 47.153752675300176,
        "pat-acc": 73.58798563499836,
        "zero_param_per": 80.91403583617748
    },
    "23.0% acc drop allowed": {
        "ptp-acc": 52.66532448217071,
        "pat-acc": 74.72838901585229,
        "zero_param_per": 77.46373720136519
    },
    "19.25% acc drop allowed": {
        "ptp-acc": 58.26078282003845,
        "pat-acc": 74.77645373091015,
        "zero_param_per": 74.36540102389078
    },
    "16.5% acc drop allowed": {
        "ptp-acc": 61.92413030072188,
        "pat-acc": 75.15643704429209,
        "zero_param_per": 68.55802047781569
    },
    "17.25% acc drop allowed": {
        "ptp-acc": 60.59464214459317,
        "pat-acc": 75.08796749737004,
        "zero_param_per": 70.74445392491468
    },
    "47.5% acc drop allowed": {
        "ptp-acc": 29.19251278702797,
        "pat-acc": 73.46147567744042,
        "zero_param_per": 87.29202218430035
    },
    "36.0% acc drop allowed": {
        "ptp-acc": 43.43735263176987,
        "pat-acc": 74.4418144883375,
        "zero_param_per": 79.82081911262799
    },
    "33.25% acc drop allowed": {
        "ptp-acc": 47.153752675300176,
        "pat-acc": 74.28401712192114,
        "zero_param_per": 80.91403583617748
    },
    "11.25% acc drop allowed": {
        "ptp-acc": 64.99301701309537,
        "pat-acc": 74.8893604672253,
        "zero_param_per": 64.19581911262799
    },
    "43.25% acc drop allowed": {
        "ptp-acc": 39.031269271230094,
        "pat-acc": 73.33405883846628,
        "zero_param_per": 86.71608361774744
    },
    "49.5% acc drop allowed": {
        "ptp-acc": 28.24391482569739,
        "pat-acc": 73.3463017375848,
        "zero_param_per": 87.8039675767918
    },
    "15.0% acc drop allowed": {
        "ptp-acc": 62.62016178764465,
        "pat-acc": 75.01451010265897,
        "zero_param_per": 73.10153583617748
    },
    "21.25% acc drop allowed": {
        "ptp-acc": 58.030434940327204,
        "pat-acc": 74.97914172742772,
        "zero_param_per": 73.26685153583618
    },
    "7.0% acc drop allowed": {
        "ptp-acc": 69.57548880908332,
        "pat-acc": 75.17638843544817,
        "zero_param_per": 68.55802047781569
    },
    "32.25% acc drop allowed": {
        "ptp-acc": 47.153752675300176,
        "pat-acc": 73.9035803678311,
        "zero_param_per": 80.91403583617748
    },
    "28.75% acc drop allowed": {
        "ptp-acc": 47.153752675300176,
        "pat-acc": 74.2214423042043,
        "zero_param_per": 80.91403583617748
    },
    "35.25% acc drop allowed": {
        "ptp-acc": 43.43735263176987,
        "pat-acc": 74.46811404940689,
        "zero_param_per": 79.82081911262799
    },
    "17.0% acc drop allowed": {
        "ptp-acc": 58.5174302608191,
        "pat-acc": 74.99501215221098,
        "zero_param_per": 70.23250853242321
    },
    "44.75% acc drop allowed": {
        "ptp-acc": 39.031269271230094,
        "pat-acc": 71.50759966626764,
        "zero_param_per": 86.71608361774744
    },
    "23.75% acc drop allowed": {
        "ptp-acc": 52.66532448217071,
        "pat-acc": 74.48987920339536,
        "zero_param_per": 77.46373720136519
    },
    "22.75% acc drop allowed": {
        "ptp-acc": 54.44099829506294,
        "pat-acc": 74.6367939928175,
        "zero_param_per": 77.29308873720136
    },
    "28.25% acc drop allowed": {
        "ptp-acc": 52.808611745928104,
        "pat-acc": 74.7020894547829,
        "zero_param_per": 77.63971843003414
    },
    "43.5% acc drop allowed": {
        "ptp-acc": 39.031269271230094,
        "pat-acc": 71.68806906808865,
        "zero_param_per": 86.71608361774744
    },
    "26.5% acc drop allowed": {
        "ptp-acc": 56.34726303188595,
        "pat-acc": 74.64994377335219,
        "zero_param_per": 68.90465017064847
    },
    "10.5% acc drop allowed": {
        "ptp-acc": 65.03382667682374,
        "pat-acc": 75.07209707258679,
        "zero_param_per": 61.326791808873715
    },
    "27.5% acc drop allowed": {
        "ptp-acc": 49.52932854499946,
        "pat-acc": 74.43410599629992,
        "zero_param_per": 72.00831911262799
    },
    "35.75% acc drop allowed": {
        "ptp-acc": 43.43735263176987,
        "pat-acc": 74.40463235027387,
        "zero_param_per": 79.82081911262799
    },
    "30.75% acc drop allowed": {
        "ptp-acc": 47.153752675300176,
        "pat-acc": 74.42503718213807,
        "zero_param_per": 80.91403583617748
    },
    "7.75% acc drop allowed": {
        "ptp-acc": 68.80826713098995,
        "pat-acc": 75.25619400007255,
        "zero_param_per": 69.65123720136519
    },
    "38.5% acc drop allowed": {
        "ptp-acc": 37.26058330612689,
        "pat-acc": 73.48505459426126,
        "zero_param_per": 85.86284129692832
    },
    "17.75% acc drop allowed": {
        "ptp-acc": 60.039177277179235,
        "pat-acc": 75.11290673631515,
        "zero_param_per": 71.08575085324233
    },
    "46.75% acc drop allowed": {
        "ptp-acc": 29.571589218993726,
        "pat-acc": 73.48414771284507,
        "zero_param_per": 86.9507252559727
    },
    "29.75% acc drop allowed": {
        "ptp-acc": 47.153752675300176,
        "pat-acc": 74.27494830775927,
        "zero_param_per": 80.91403583617748
    },
    "42.75% acc drop allowed": {
        "ptp-acc": 39.031269271230094,
        "pat-acc": 73.0216381905902,
        "zero_param_per": 86.71608361774744
    },
    "27.25% acc drop allowed": {
        "ptp-acc": 49.52932854499946,
        "pat-acc": 74.26678637501361,
        "zero_param_per": 72.00831911262799
    },
    "7.25% acc drop allowed": {
        "ptp-acc": 69.57548880908332,
        "pat-acc": 75.1940726230638,
        "zero_param_per": 68.55802047781569
    },
    "25.25% acc drop allowed": {
        "ptp-acc": 50.34642870098306,
        "pat-acc": 74.90024304421954,
        "zero_param_per": 65.86497440273038
    },
    "36.5% acc drop allowed": {
        "ptp-acc": 43.43735263176987,
        "pat-acc": 74.50801683171909,
        "zero_param_per": 79.82081911262799
    },
    "28.0% acc drop allowed": {
        "ptp-acc": 47.70831066129793,
        "pat-acc": 74.9093118583814,
        "zero_param_per": 67.8754266211604
    },
    "28.5% acc drop allowed": {
        "ptp-acc": 47.153752675300176,
        "pat-acc": 74.21917510066383,
        "zero_param_per": 80.91403583617748
    },
    "46.5% acc drop allowed": {
        "ptp-acc": 29.571589218993726,
        "pat-acc": 73.06924946493996,
        "zero_param_per": 86.9507252559727
    },
    "41.5% acc drop allowed": {
        "ptp-acc": 39.031269271230094,
        "pat-acc": 73.40026118184787,
        "zero_param_per": 86.71608361774744
    },
    "5.25% acc drop allowed": {
        "ptp-acc": 70.91540610149816,
        "pat-acc": 75.40991040011608,
        "zero_param_per": 67.47013651877133
    },
    "40.25% acc drop allowed": {
        "ptp-acc": 35.915224725214934,
        "pat-acc": 73.53901403852433,
        "zero_param_per": 86.37478668941979
    },
    "26.0% acc drop allowed": {
        "ptp-acc": 56.34726303188595,
        "pat-acc": 74.71795987956615,
        "zero_param_per": 68.90465017064847
    },
    "11.5% acc drop allowed": {
        "ptp-acc": 64.99301701309537,
        "pat-acc": 75.13149780534697,
        "zero_param_per": 64.19581911262799
    },
    "35.0% acc drop allowed": {
        "ptp-acc": 43.43735263176987,
        "pat-acc": 74.2214423042043,
        "zero_param_per": 79.82081911262799
    },
    "0.5% acc drop allowed": {
        "ptp-acc": 75.038995900896,
        "pat-acc": 75.37318170276055,
        "zero_param_per": 51.08788395904437
    },
    "45.25% acc drop allowed": {
        "ptp-acc": 39.031269271230094,
        "pat-acc": 72.11248957086372,
        "zero_param_per": 86.71608361774744
    },
    "10.75% acc drop allowed": {
        "ptp-acc": 65.70219828055284,
        "pat-acc": 75.17684187615627,
        "zero_param_per": 62.00938566552902
    },
    "41.25% acc drop allowed": {
        "ptp-acc": 39.031269271230094,
        "pat-acc": 73.27193746145754,
        "zero_param_per": 86.71608361774744
    },
    "5.5% acc drop allowed": {
        "ptp-acc": 70.91540610149816,
        "pat-acc": 75.35821815939347,
        "zero_param_per": 67.47013651877133
    },
    "34.0% acc drop allowed": {
        "ptp-acc": 43.43735263176987,
        "pat-acc": 74.40599267239816,
        "zero_param_per": 79.82081911262799
    },
    "9.0% acc drop allowed": {
        "ptp-acc": 67.56357238727465,
        "pat-acc": 75.30153807088185,
        "zero_param_per": 70.74445392491468
    },
    "31.75% acc drop allowed": {
        "ptp-acc": 47.153752675300176,
        "pat-acc": 73.69273043856785,
        "zero_param_per": 80.91403583617748
    },
    "7.5% acc drop allowed": {
        "ptp-acc": 68.80826713098995,
        "pat-acc": 75.35277687089636,
        "zero_param_per": 69.65123720136519
    },
    "25.5% acc drop allowed": {
        "ptp-acc": 56.34726303188595,
        "pat-acc": 74.69800848841005,
        "zero_param_per": 68.90465017064847
    },
    "1.5% acc drop allowed": {
        "ptp-acc": 74.50212210251388,
        "pat-acc": 75.22173250625748,
        "zero_param_per": 58.73506825938567
    },
    "5.0% acc drop allowed": {
        "ptp-acc": 70.91540610149816,
        "pat-acc": 75.35186998948018,
        "zero_param_per": 67.47013651877133
    },
    "15.5% acc drop allowed": {
        "ptp-acc": 60.00018137628324,
        "pat-acc": 74.76239706895926,
        "zero_param_per": 73.51215870307168
    },
    "45.5% acc drop allowed": {
        "ptp-acc": 39.031269271230094,
        "pat-acc": 72.99715239235317,
        "zero_param_per": 86.71608361774744
    },
    "6.5% acc drop allowed": {
        "ptp-acc": 70.12505894729205,
        "pat-acc": 75.31106032575181,
        "zero_param_per": 67.46480375426621
    },
    "13.5% acc drop allowed": {
        "ptp-acc": 62.62016178764465,
        "pat-acc": 75.06076105488447,
        "zero_param_per": 73.10153583617748
    },
    "0.25% acc drop allowed": {
        "ptp-acc": 75.23034787971125,
        "pat-acc": 75.41580512932128,
        "zero_param_per": 48.730802047781566
    },
    "24.5% acc drop allowed": {
        "ptp-acc": 51.037018899408714,
        "pat-acc": 74.97868828671963,
        "zero_param_per": 63.678540955631405
    },
    "13.0% acc drop allowed": {
        "ptp-acc": 62.62016178764465,
        "pat-acc": 75.14374070446549,
        "zero_param_per": 73.10153583617748
    },
    "27.75% acc drop allowed": {
        "ptp-acc": 47.788569666630394,
        "pat-acc": 74.83857510791889,
        "zero_param_per": 67.7047781569966
    },
    "23.25% acc drop allowed": {
        "ptp-acc": 52.66532448217071,
        "pat-acc": 74.50892371313527,
        "zero_param_per": 77.46373720136519
    },
    "38.0% acc drop allowed": {
        "ptp-acc": 46.853121485834514,
        "pat-acc": 72.88923350382704,
        "zero_param_per": 85.62286689419795
    },
    "2.25% acc drop allowed": {
        "ptp-acc": 73.61428519606777,
        "pat-acc": 75.37272826205245,
        "zero_param_per": 59.82295221843004
    },
    "43.0% acc drop allowed": {
        "ptp-acc": 39.031269271230094,
        "pat-acc": 73.14860158885624,
        "zero_param_per": 86.71608361774744
    },
    "14.75% acc drop allowed": {
        "ptp-acc": 62.62016178764465,
        "pat-acc": 75.09612943011572,
        "zero_param_per": 73.10153583617748
    },
    "46.25% acc drop allowed": {
        "ptp-acc": 29.571589218993726,
        "pat-acc": 73.62562121377009,
        "zero_param_per": 86.9507252559727
    },
    "41.75% acc drop allowed": {
        "ptp-acc": 39.031269271230094,
        "pat-acc": 73.1381724525701,
        "zero_param_per": 86.71608361774744
    },
    "2.5% acc drop allowed": {
        "ptp-acc": 73.43109514999819,
        "pat-acc": 75.36411288859868,
        "zero_param_per": 56.54863481228669
    },
    "12.0% acc drop allowed": {
        "ptp-acc": 63.54518083215439,
        "pat-acc": 75.16006456995683,
        "zero_param_per": 65.69432593856655
    },
    "16.25% acc drop allowed": {
        "ptp-acc": 59.29417419378242,
        "pat-acc": 75.0838865309972,
        "zero_param_per": 68.04607508532423
    },
    "11.75% acc drop allowed": {
        "ptp-acc": 63.74424130300722,
        "pat-acc": 75.17230746907535,
        "zero_param_per": 63.678540955631405
    },
    "35.5% acc drop allowed": {
        "ptp-acc": 43.43735263176987,
        "pat-acc": 74.46539340515834,
        "zero_param_per": 79.81548634812286
    },
    "19.75% acc drop allowed": {
        "ptp-acc": 55.843490405194615,
        "pat-acc": 75.0126963398266,
        "zero_param_per": 61.321459044368595
    },
    "20.5% acc drop allowed": {
        "ptp-acc": 55.05450357311278,
        "pat-acc": 75.08796749737004,
        "zero_param_per": 65.68899317406144
    },
    "0.75% acc drop allowed": {
        "ptp-acc": 74.89525519643051,
        "pat-acc": 75.3885986868357,
        "zero_param_per": 55.4554180887372
    },
    "48.5% acc drop allowed": {
        "ptp-acc": 29.19251278702797,
        "pat-acc": 73.15449631806145,
        "zero_param_per": 87.29202218430035
    },
    "14.25% acc drop allowed": {
        "ptp-acc": 62.62016178764465,
        "pat-acc": 75.05940073276018,
        "zero_param_per": 73.10153583617748
    },
    "3.75% acc drop allowed": {
        "ptp-acc": 72.14150977618166,
        "pat-acc": 75.38995900895999,
        "zero_param_per": 53.274317406143346
    },
    "3.0% acc drop allowed": {
        "ptp-acc": 73.04068270033011,
        "pat-acc": 75.29700366380092,
        "zero_param_per": 63.1026023890785
    },
    "44.25% acc drop allowed": {
        "ptp-acc": 39.031269271230094,
        "pat-acc": 73.2166176950702,
        "zero_param_per": 86.71608361774744
    },
    "37.75% acc drop allowed": {
        "ptp-acc": 38.43771538433634,
        "pat-acc": 74.20058403163202,
        "zero_param_per": 81.32465870307168
    },
    "33.5% acc drop allowed": {
        "ptp-acc": 43.43735263176987,
        "pat-acc": 74.32391990423332,
        "zero_param_per": 79.82081911262799
    },
    "9.75% acc drop allowed": {
        "ptp-acc": 67.56357238727465,
        "pat-acc": 75.22762723546269,
        "zero_param_per": 70.74445392491468
    },
    "22.25% acc drop allowed": {
        "ptp-acc": 59.12232016541517,
        "pat-acc": 74.99002430442195,
        "zero_param_per": 75.45328498293516
    },
    "24.75% acc drop allowed": {
        "ptp-acc": 50.754071897558674,
        "pat-acc": 74.7342837450575,
        "zero_param_per": 65.52367747440273
    },
    "19.0% acc drop allowed": {
        "ptp-acc": 58.26078282003845,
        "pat-acc": 74.76466427249973,
        "zero_param_per": 74.36540102389078
    },
    "22.5% acc drop allowed": {
        "ptp-acc": 53.073874560162515,
        "pat-acc": 74.72657525301992,
        "zero_param_per": 76.95179180887372
    },
    "48.75% acc drop allowed": {
        "ptp-acc": 29.19251278702797,
        "pat-acc": 72.88968694453513,
        "zero_param_per": 87.29202218430035
    },
    "3.5% acc drop allowed": {
        "ptp-acc": 72.66523379402909,
        "pat-acc": 75.30017774875758,
        "zero_param_per": 62.00405290102389
    },
    "24.25% acc drop allowed": {
        "ptp-acc": 51.357148039322375,
        "pat-acc": 74.94513367432074,
        "zero_param_per": 63.50789249146758
    },
    "0.0% acc drop allowed": {
        "ptp-acc": 75.48155403199478,
        "pat-acc": 75.40401567091087,
        "zero_param_per": 2.186433447098976
    },
    "8.75% acc drop allowed": {
        "ptp-acc": 67.56357238727465,
        "pat-acc": 75.32602386911887,
        "zero_param_per": 70.74445392491468
    },
    "11.0% acc drop allowed": {
        "ptp-acc": 64.99301701309537,
        "pat-acc": 75.0131497805347,
        "zero_param_per": 64.19581911262799
    },
    "30.0% acc drop allowed": {
        "ptp-acc": 47.153752675300176,
        "pat-acc": 74.1692966227736,
        "zero_param_per": 80.91403583617748
    },
    "46.0% acc drop allowed": {
        "ptp-acc": 39.031269271230094,
        "pat-acc": 72.70060216926035,
        "zero_param_per": 86.71608361774744
    },
    "8.0% acc drop allowed": {
        "ptp-acc": 68.80826713098995,
        "pat-acc": 75.30289839300613,
        "zero_param_per": 69.65123720136519
    },
    "6.25% acc drop allowed": {
        "ptp-acc": 70.12505894729205,
        "pat-acc": 75.27977291689339,
        "zero_param_per": 67.46480375426621
    },
    "26.75% acc drop allowed": {
        "ptp-acc": 56.34726303188595,
        "pat-acc": 74.77418652736968,
        "zero_param_per": 68.90465017064847
    },
    "39.75% acc drop allowed": {
        "ptp-acc": 37.26058330612689,
        "pat-acc": 73.73580730583669,
        "zero_param_per": 85.86284129692832
    },
    "39.0% acc drop allowed": {
        "ptp-acc": 37.26058330612689,
        "pat-acc": 73.37078753582182,
        "zero_param_per": 85.86284129692832
    },
    "29.5% acc drop allowed": {
        "ptp-acc": 47.153752675300176,
        "pat-acc": 74.39193601044727,
        "zero_param_per": 80.91403583617748
    },
    "1.25% acc drop allowed": {
        "ptp-acc": 74.23549896615519,
        "pat-acc": 75.3115137664599,
        "zero_param_per": 58.05247440273038
    },
    "18.0% acc drop allowed": {
        "ptp-acc": 60.039177277179235,
        "pat-acc": 75.00952225486995,
        "zero_param_per": 71.08575085324233
    },
    "38.25% acc drop allowed": {
        "ptp-acc": 46.853121485834514,
        "pat-acc": 73.39572677476693,
        "zero_param_per": 85.62286689419795
    },
    "31.0% acc drop allowed": {
        "ptp-acc": 47.153752675300176,
        "pat-acc": 74.33661624405993,
        "zero_param_per": 80.91403583617748
    },
    "3.25% acc drop allowed": {
        "ptp-acc": 72.66523379402909,
        "pat-acc": 75.35595095585302,
        "zero_param_per": 62.00405290102389
    },
    "1.75% acc drop allowed": {
        "ptp-acc": 73.74034171291761,
        "pat-acc": 75.39766750099757,
        "zero_param_per": 54.60217576791809
    },
    "9.5% acc drop allowed": {
        "ptp-acc": 67.56357238727465,
        "pat-acc": 75.31468785141655,
        "zero_param_per": 70.74445392491468
    },
    "40.75% acc drop allowed": {
        "ptp-acc": 35.915224725214934,
        "pat-acc": 73.4696376101861,
        "zero_param_per": 86.37478668941979
    },
    "4.25% acc drop allowed": {
        "ptp-acc": 71.82863568759748,
        "pat-acc": 75.36456632930678,
        "zero_param_per": 65.28370307167235
    },
    "48.25% acc drop allowed": {
        "ptp-acc": 29.19251278702797,
        "pat-acc": 73.17082018355279,
        "zero_param_per": 87.29202218430035
    },
    "18.25% acc drop allowed": {
        "ptp-acc": 61.80442195378532,
        "pat-acc": 74.71025138752857,
        "zero_param_per": 73.2721843003413
    },
    "34.75% acc drop allowed": {
        "ptp-acc": 43.43735263176987,
        "pat-acc": 74.15932092719557,
        "zero_param_per": 79.82081911262799
    },
    "29.25% acc drop allowed": {
        "ptp-acc": 47.153752675300176,
        "pat-acc": 74.371077737875,
        "zero_param_per": 80.91403583617748
    },
    "19.5% acc drop allowed": {
        "ptp-acc": 56.03212173976131,
        "pat-acc": 75.0267530017775,
        "zero_param_per": 59.140358361774744
    },
    "20.75% acc drop allowed": {
        "ptp-acc": 61.0598723110966,
        "pat-acc": 74.89978960351145,
        "zero_param_per": 73.26685153583618
    },
    "32.5% acc drop allowed": {
        "ptp-acc": 47.153752675300176,
        "pat-acc": 74.25681067943556,
        "zero_param_per": 80.91403583617748
    },
    "43.75% acc drop allowed": {
        "ptp-acc": 39.031269271230094,
        "pat-acc": 73.06607537998332,
        "zero_param_per": 86.71608361774744
    },
    "18.75% acc drop allowed": {
        "ptp-acc": 59.0751623317735,
        "pat-acc": 74.64813001051982,
        "zero_param_per": 74.02410409556313
    },
    "22.0% acc drop allowed": {
        "ptp-acc": 59.12232016541517,
        "pat-acc": 74.91702035041898,
        "zero_param_per": 75.45328498293516
    },
    "41.0% acc drop allowed": {
        "ptp-acc": 39.031269271230094,
        "pat-acc": 71.93700801683173,
        "zero_param_per": 86.71608361774744
    },
    "27.0% acc drop allowed": {
        "ptp-acc": 49.52932854499946,
        "pat-acc": 74.31802517502811,
        "zero_param_per": 72.00831911262799
    },
    "12.75% acc drop allowed": {
        "ptp-acc": 63.69436282511699,
        "pat-acc": 75.07617803895963,
        "zero_param_per": 72.93088737201366
    },
    "44.0% acc drop allowed": {
        "ptp-acc": 39.031269271230094,
        "pat-acc": 73.36398592520042,
        "zero_param_per": 86.71608361774744
    },
    "6.0% acc drop allowed": {
        "ptp-acc": 69.49568324445896,
        "pat-acc": 75.29382957884427,
        "zero_param_per": 66.78220989761093
    },
    "15.75% acc drop allowed": {
        "ptp-acc": 59.94486160989589,
        "pat-acc": 75.20042079297711,
        "zero_param_per": 63.678540955631405
    },
    "20.0% acc drop allowed": {
        "ptp-acc": 55.567345013965976,
        "pat-acc": 75.0199513911561,
        "zero_param_per": 63.50789249146758
    },
    "42.0% acc drop allowed": {
        "ptp-acc": 39.031269271230094,
        "pat-acc": 73.34720861900098,
        "zero_param_per": 86.71608361774744
    },
    "8.25% acc drop allowed": {
        "ptp-acc": 67.56357238727465,
        "pat-acc": 75.28884173105524,
        "zero_param_per": 70.74445392491468
    },
    "4.75% acc drop allowed": {
        "ptp-acc": 71.587405230892,
        "pat-acc": 75.31106032575181,
        "zero_param_per": 66.37691979522185
    },
    "47.25% acc drop allowed": {
        "ptp-acc": 29.19251278702797,
        "pat-acc": 73.46147567744042,
        "zero_param_per": 87.29202218430035
    },
    "38.75% acc drop allowed": {
        "ptp-acc": 37.26058330612689,
        "pat-acc": 73.65010701200711,
        "zero_param_per": 85.86284129692832
    },
    "1.0% acc drop allowed": {
        "ptp-acc": 74.65357129901695,
        "pat-acc": 75.30743280008706,
        "zero_param_per": 56.54863481228669
    },
    "48.0% acc drop allowed": {
        "ptp-acc": 29.19251278702797,
        "pat-acc": 73.56758080313418,
        "zero_param_per": 87.29202218430035
    },
    "44.5% acc drop allowed": {
        "ptp-acc": 39.031269271230094,
        "pat-acc": 72.60719338339318,
        "zero_param_per": 86.71608361774744
    },
    "34.5% acc drop allowed": {
        "ptp-acc": 43.43735263176987,
        "pat-acc": 74.41460804585192,
        "zero_param_per": 79.82081911262799
    }
}

# Extract data and sort by zero_param_per
sorted_data = sorted(data.items(), key=lambda x: x[1]['zero_param_per'])

keys = [item[0] for item in sorted_data]
zero_param_per = [item[1]['zero_param_per'] for item in sorted_data]
ptp_acc = [item[1]['ptp-acc'] for item in sorted_data][-10:]
pat_acc = [item[1]['pat-acc'] for item in sorted_data][-10:]


# Create new labels combining the accuracy drop allowed with zero_param_per
labels_with_zero_param = [f"{key}\n({zero_param_per[i]:.1f}% zero params)" for i, key in enumerate(keys)][-10:]

# Plotting the simulated data
plt.figure(figsize=(16, 10))  # Make the plot much wider

# Plotting the points without connecting lines
plt.scatter(labels_with_zero_param, ptp_acc, marker='o', label='Post Training Pruning (ptp-acc)', color='blue', s=40)
plt.scatter(labels_with_zero_param, pat_acc, marker='o', label='Pruning Aware Training (pat-acc)', color='orange', s=80)

# Connect vertically
for i in range(len(labels_with_zero_param)):
    plt.plot([labels_with_zero_param[i], labels_with_zero_param[i]], [ptp_acc[i], pat_acc[i]], color='gray', linestyle='--', markersize=50)

plt.xlabel('Accuracy Drop Allowed and Zero Parameter Percentage', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Accuracy vs Accuracy Drop Allowed and Zero Parameter Percentage', fontsize=18)
plt.xticks(rotation=90, ha='center', fontsize=14)  # Rotate labels 90 degrees and reduce font size
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_vs_zero_params.png')  # Save the figure
plt.show()