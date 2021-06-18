from scipy.io.wavfile import read, write
from math import log10, sqrt, ceil
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time
from PIL import Image
import random
import difflib


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_value = original.max()
    psnr = 20 * log10(max_value / sqrt(mse))
    return psnr


def word_to_bits(enc_word):
    if len(enc_word) == 0:
        raise ValueError('Nothing to embed')
    result = list(map(ord, enc_word))
    max_len = len(bin(max(result))[2:])
    encode_str = ''
    for value in result:
        encode_str += '{:0{}b}'.format(value, max_len)
    encode_str += '{:0{}b}'.format(0, max_len)
    return encode_str, max_len


def audio_lsb_embed(audio, message):
    if audio.shape[0] <= len(message):
        raise ValueError('Message is too long.\nMsg length: {: >16d}\nContainer length: {: >10d}'.format(len(message), audio.shape[0]))
    bits_iter = iter(message)
    new_audio = np.copy(audio)
    for i in range(len(audio)):
        try:
            val = next(bits_iter)
            if val == '0':
                new_audio[i] = audio[i] & ~1
            else:
                new_audio[i] = audio[i] | 1
        except StopIteration:
        #     print('Error: {}'.format(E))
            return new_audio


def create_str(encode_str, max_len):
    result = ['0b' + encode_str[i:i + max_len] for i in range(0, len(encode_str), max_len)]
    decoded_word = ''
    for letter in result:
        let = chr(int(letter, 2))
        if let == chr(0):
            break
        decoded_word += let
    return decoded_word


def audio_lsb_extract(audio, max_len):
    encode_str = ''
    for i in audio:
        encode_str += bin(i)[-1]
        if len(encode_str) >= max_len and not len(encode_str) % max_len and encode_str[-max_len:] == '{:0{}b}'.format(0, max_len):
            decoded_word = create_str(encode_str, max_len)
            return decoded_word


def LSB():
    freq, sample = read('dance_20_sec.wav')

    # msg = ''
    # data = [[], []]
    # for i in range(0, 1000):
    #     msg += chr(random.randint(31, 90))
    #     try:
    #         bits, max_len = word_to_bits(msg)
    #         sample_with_lsb = audio_lsb_embed(sample, bits)
    #         msg = audio_lsb_extract(sample_with_lsb, max_len)
    #         psnr = PSNR(sample, sample_with_lsb)
    #         data[0].append(msg)
    #         data[1].append(psnr)
    #         if not i % 10:
    #             print('Input word:  {}'.format(msg))
    #             print('Output word: {}'.format(msg))
    #             print('PSNR:        {}\n'.format(psnr))
    #     except Exception as e:
    #         print(e)
    #         break
    #
    # fig, ax = plt.subplots()
    # ax.plot([len(word) for word in data[0]], data[1])
    # ax.set_ylabel('PSNR')
    # ax.set_xlabel('word length')
    # plt.show()

    # msg = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. '

    msg = ''
    for _ in range(10_000):
        msg += chr(random.randint(0x41, 0x5a))
    print(len(msg))

    # msg = 'Secret word'
    # print('Input word:  {}'.format(msg))

    start_time = time.time()

    bits, max_len = word_to_bits(msg)

    # print('Sample len:  {: >10d}\n'
    #       'Bits:        {: >10d}\n'
    #       'len / 7      {: >10d}'.format(sample.shape[0], len(bits), sample.shape[0] // 7))

    sample_with_lsb = audio_lsb_embed(sample, bits)

    # sample_with_lsb = np.array([bin(i)[-1] for i in sample_with_lsb])
    # one_side = ceil(sqrt(sample_with_lsb.shape[0]))

    # attack_array = np.zeros(one_side ** 2, np.uint8)
    # attack_array[:sample_with_lsb.shape[0]] = sample_with_lsb
    # attack_array = attack_array.reshape(one_side, one_side)
    # attack_array[attack_array == 1] = 255
    #
    # img = Image.fromarray(attack_array)
    # img.save('Lsb attack.bmp')

    # sample_with_lsb, noise = insert_noise(sample_with_lsb, 0.00003)

    # fig, ax = plt.subplots(2, 1, figsize=(9, 3), sharey=True, sharex=True)
    # ax[0].plot(noise)
    # ax[1].plot(sample_with_lsb)
    # plt.show()
    #
    # fig, ax = plt.subplots(1, 1, figsize=(9, 3), sharey=True, sharex=True)
    # ax.plot(noise)
    # plt.show()

    word = audio_lsb_extract(sample_with_lsb, max_len)

    print("%s sec." % (time.time() - start_time))
    print('Output word: {}'.format(word))
    # print(word_noise)
    print('PSNR: {}'.format(PSNR(sample, sample_with_lsb)))
    # print('PSNR: {}'.format(PSNR(sample, sample_with_lsb_noise)))


def audio_phase_embed(audio, message, dtype=np.int16, show=False, blocks=4):
    sample = np.copy(audio)
    sample_len = sample.shape[0]
    bits, max_len = word_to_bits(message)
    # print('Bits len: {: >13d}'.format(len(bits)))
    bits_array = np.array([int(x) for x in bits])
    bits_array[bits_array == 0] = -1
    bits_array_pi = bits_array * (-np.pi / 2)

    msg_len = (len(message) + 1) * max_len
    seg_len = int(2 * 2 ** np.ceil(np.log2(2 * msg_len)))

    if sample_len < seg_len:
        raise ValueError('Message is too long.\nSample length:  {: >10d}\nSegment length: {: >10d}'.format(sample_len, seg_len))

    seg_len_half = seg_len // 2
    seg_num = int(np.ceil(sample.shape[0] / seg_len))
    # print('Sample len:  {: >10d}\n'
    #       'Segment len: {: >10d}\n'
    #       'Seg_num :    {: >10d}'.format(sample_len, seg_len, seg_num))
    if seg_num < blocks:
        show = False

    new_sample = np.zeros(seg_len * seg_num, dtype=dtype)
    new_sample[:len(sample)] = sample.astype(dtype=dtype)
    new_sample_spitted = np.split(new_sample, seg_num)
    if seg_len < len(new_sample_spitted[0]):
        raise ValueError('Word is too long {} {}'.format(seg_num, len(new_sample_spitted[0])))

    if new_sample_spitted[0].sum() == 0:
        raise ValueError("First sample don't have data")

    fft_array = np.fft.fft(new_sample_spitted)
    a_magnitude = np.abs(fft_array)
    phi_phase = np.angle(fft_array)
    diff_phase = np.diff(phi_phase, axis=0)

    if show:
        fig, ax = plt.subplots(4, blocks, figsize=(9, 9), sharey=True, sharex=True)
        ax[0][0].set_ylabel('Phi')
        for i in range(blocks):
            ax[0][i].plot(phi_phase[i])

    phi_phase[0][seg_len_half - msg_len: seg_len_half] = bits_array_pi = bits_array_pi
    phi_phase[0][seg_len_half + 1: seg_len_half + 1 + msg_len] = -bits_array_pi[::-1]

    if show:
        ax[2][0].set_ylabel('Insert data')
        ax[1][0].set_ylabel('Diff')
        for i in range(blocks):
            ax[2][i].plot(phi_phase[i])
        for i in range(blocks - 1):
            ax[1][i + 1].plot(diff_phase[i])

    # for i in range(1, len(phi_phase)):
    #     phi_phase[i] = phi_phase[i - 1] + diff_phase[i - 1]

    if show:
        ax[3][0].set_ylabel('Add diff')
        for i in range(blocks):
            ax[3][i].set_xlabel(str(i))
            ax[3][i].plot(phi_phase[i])
        plt.show()

    blocks = np.fft.ifft(a_magnitude * np.exp(1j * phi_phase)).real

    return blocks.flatten().astype(dtype)[:sample_len], new_sample[:sample_len], seg_len, msg_len, max_len


def audio_phase_extract(audio, seg_len, msg_len, max_len):
    sample = np.copy(audio)
    seg_len_half = seg_len // 2

    segment = sample[:seg_len]

    # plt.plot(segment)
    # plt.show()

    segment_phase = np.angle(np.fft.fft(segment))

    # plt.plot(segment_phase[seg_len_half - msg_len:seg_len_half])
    # plt.show()

    encode_str = ''
    for each in segment_phase[seg_len_half - msg_len:seg_len_half]:
        if each < 0:
            encode_str += '1'
        else:
            encode_str += '0'
    decoded_word = create_str(encode_str, max_len)
    return decoded_word


def insert_noise(sample, percent=0.001):
    noise = (np.random.randn(sample.shape[0]) * (np.max(sample) * percent)).astype(np.int16)
    return sample + noise, noise


def phase():
    name = 'dance_20_sec.wav'
    new_name = 'steg_' + name
    dtype = np.int16

    msg = ''
    for _ in range(50):
        msg += chr(random.randint(0x41, 0x5a))

    # msg = 'Secret word'

    # msg = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. '

    print('Input word:  {}'.format(msg))

    # msg = 'Secret word'
    # print(len(msg))
    # print('Input word:  {}'.format(msg))

    freq, sample = read(name)

    # fig, ax = plt.subplots(1, 1, figsize=(9, 3), sharey=True, sharex=True)
    # ax.plot(sample)
    # plt.show()

    start_time = time.time()

    new_audio, old_audio, seg_len, msg_len, max_len = audio_phase_embed(sample.astype(dtype), msg, dtype=dtype)

    # print(len(new_audio))
    # plt.plot(new_audio[:seg_len])
    # plt.show()
    # new_audio = np.append(insert_noise(new_audio[:seg_len], 0.01)[0], new_audio[seg_len:])
    # plt.plot(new_audio[:seg_len])
    # plt.show()
    # print(len(new_audio))

    # new_audio, noise = insert_noise(new_audio, 0.001)
    # new_audio[:seg_len] = new_audio_seg

    # fig, ax = plt.subplots(1, 1, figsize=(9, 3), sharey=True, sharex=True)
    # ax.plot(noise)
    # plt.show()

    write(new_name, freq, new_audio.astype(np.int16))

    # new_name = 'joji_5.wav'
    # freq, new_audio = read(new_name)

    ext_msg = audio_phase_extract(new_audio, seg_len, msg_len, max_len)

    # msg_diff = [li for li in difflib.ndiff(msg, ext_msg) if li[0] != ' ']

    print("--- %s seconds ---" % (time.time() - start_time))

    print('Output word: {}\n'.format(ext_msg))
    print('PSNR:        {}\n'.format(PSNR(old_audio, new_audio)))

    # f, t, Sxx = signal.spectrogram(new_audio, fs=freq)
    # dBS = 10 * np.log10(Sxx)
    # plt.pcolormesh(t, f, dBS)
    # plt.show()

    # fig, ax = plt.subplots(1, 2, figsize=(6, 3), sharey=True, sharex=True)
    #
    # f, t, Sxx = signal.spectrogram(old_audio, fs=freq)
    # dBS = 10 * np.log10(Sxx)
    # ax[0].pcolormesh(t, f, dBS)
    #
    # f, t, Sxx = signal.spectrogram(new_audio, fs=freq)
    # dBS = 10 * np.log10(Sxx)
    # ax[1].pcolormesh(t, f, dBS)
    # plt.show()

    # for _ in range(50):
    #     new_audio, noise = insert_noise(new_audio, 0.001)
    #
    #     # write('noise.wav', freq, noise)
    #
    #     # write(new_name, freq, new_audio.astype(np.int16))
    #
    #     # freq, sample = read(new_name)
    #
    #     msg = audio_phase_extract(new_audio, seg_len, msg_len, max_len)
    #     print('Output word: {}'.format(msg))
    #     print('PSNR:        {}'.format(PSNR(old_audio, new_audio)))

    # msg = ''
    # data = [[], []]
    # for i in range(0, 1000):
    #     msg += chr(random.randint(31, 90))
    #     try:
    #         new_audio, old_audio, seg_len, msg_len, max_len = audio_phase_embed(sample, msg, dtype)
    #         msg_decode = audio_phase_extract(new_audio, seg_len, msg_len, max_len)
    #         psnr = PSNR(old_audio, new_audio)
    #         data[0].append(msg)
    #         data[1].append(psnr)
    #         if not i % 10:
    #             print('Input word:  {}'.format(msg))
    #             print('Output word: {}'.format(msg_decode))
    #             print('PSNR:        {}\n'.format(psnr))
    #     except Exception as e:
    #         print(e)
    #         break
    #
    # write(new_name, freq, new_audio.astype(np.int16))
    #
    # fig, ax = plt.subplots()
    # ax.plot([len(word) for word in data[0]], data[1])
    # ax.set_ylabel('PSNR')
    # ax.set_xlabel('word length')
    # plt.show()


def main():
    # LSB()
    phase()


if __name__ == '__main__':
    main()