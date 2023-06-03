import os
import numpy as np
import librosa
import pyworld
import argparse

import multiprocessing


def compute_f0(paths):
    try:
        path, save = paths[0], paths[1]
        x, sr = librosa.load(path, sr=16000)
        assert sr == 16000
        f0, t = pyworld.dio(
            x.astype(np.double),
            fs=sr,
            f0_ceil=900,
            frame_period=1000 * 160 / sr,
        )
        f0 = pyworld.stonemask(x.astype(np.double), f0, t, fs=16000)
        for index, pitch in enumerate(f0):
            f0[index] = round(pitch, 1)
        np.save(save, f0, allow_pickle=False)
    except:
        print(paths)
        exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-p", "--pit", help="pit", dest="pit")
    args = parser.parse_args()
    print(args.wav)
    print(args.pit)
    os.makedirs(args.pit)
    wavPath = args.wav
    pitPath = args.pit

    paths = list()
    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{pitPath}/{spks}")
            print(f">>>>>>>>>>{spks}<<<<<<<<<<")
            for file in os.listdir(f"./{wavPath}/{spks}"):
                if file.endswith(".wav"):
                    # print(file)
                    file = file[:-4]
                    path = list()
                    path.append(f"{wavPath}/{spks}/{file}.wav")
                    path.append(f"{pitPath}/{spks}/{file}.pit")
                    paths.append(path)
                    #compute_f0(f"{wavPath}/{spks}/{file}.wav", f"{pitPath}/{spks}/{file}.pit")
        else:
            file = spks
            if file.endswith(".wav"):
                # print(file)
                file = file[:-4]
                path = list()
                path.append(f"{wavPath}/{spks}/{file}.wav")
                path.append(f"{pitPath}/{spks}/{file}.pit")
                paths.append(path)
                #compute_f0(f"{wavPath}/{file}.wav", f"{pitPath}/{file}.pit")
                
    pool_obj = multiprocessing.Pool()
    ans = pool_obj.map(compute_f0, paths)
    pool_obj.close()
