import librosa.util
import librosa, librosa.display
from functions_git import *

def seg_librosa():

    # -------------------------------------------- wczytanie pliku ----------------------------------------------------
    audio_file = filedialog.askopenfilename(title="Wybierz plik", filetypes = (("wav mono files",".wav"), ("all files", "*.*")))

    # audio_file = 'C:\\Users\\konie\\PycharmProjects\\2020-po-mgr\\Niska_skad_wiadomo.wav'

    y, sr = librosa.load(audio_file)
    name_sound = Tonacja(audio_file)

    # główny słownik
    nutyLib = {}

    # początki nut - wykrywanie miejsc peaks
    hop_length = 256 # 100 - 300
    nutyLib['onset_env'] = librosa.onset.onset_strength(y, sr=sr, hop_length=hop_length)

    nutyLib['onset_samples'] = librosa.onset.onset_detect(y, sr=sr, units='samples',
                                               hop_length=hop_length, backtrack=False,
                                               pre_max=20, post_max=20,
                                               pre_avg=100,post_avg=100,
                                               delta=0.1,wait=0)

    # sample, ale z granicami
    nutyLib['onset_boundaries'] = np.concatenate([[0], nutyLib['onset_samples'], [len(y)]])

    # "kliknięcia"
    nutyLib['onset_times'] = librosa.samples_to_time(nutyLib['onset_boundaries'], sr=sr)

    plt.subplot(211)
    plt.plot(nutyLib['onset_env'])
    plt.ylabel('Siła zmian energii',fontsize=15)
    plt.xlabel('Liczba próbek',fontsize=15)
    plt.title('onset env')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(0, len(nutyLib['onset_env']))

    plt.subplot(212)
    librosa.display.waveplot(y)
    plt.vlines(nutyLib['onset_times'], -1, 1, color='r')
    plt.ylabel('Amplituda',fontsize=10)
    plt.xlabel('Czas',fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.show()


    # pitch
    def estimate_pitch(segment, sr, fmin=50.0, fmax=2000.0):

        r = librosa.autocorrelate(segment)

        # zakres auto
        i_min = sr / fmax
        i_max = sr / fmin

        # append, del
        r[:int(i_min)] = 0
        r[int(i_max):] = 0

        # max
        i = r.argmax()
        f0 = float(sr) / i
        return f0

    def estimate_pitch_and_generate_sine(y, onset_samples, sr):
        for i in range(len(nutyLib['onset_boundaries']) - 1):
            n0 = onset_samples[i]
            n1 = onset_samples[i+1]
            f0 = estimate_pitch(y[n0:n1], sr)
            nuty_hz.append(f0)

        return nuty_hz

    nuty_hz = []
    nutyLib['nameLily'] = estimate_pitch_and_generate_sine(y, nutyLib['onset_boundaries'], sr=sr)
    print('nuty', nutyLib['nameLily'])

    # pierwszy element
    nutyLib['nameLily'].pop(0)

    nutyLib['midi'] = f_to_midi(nutyLib['nameLily'])
    nutyLib['nameLily'] = nazwy_nutek(nutyLib['midi'], name_sound)

    nutyLib['dlugosci'] = Siatka_rytmiczna(nutyLib)

    nutyLib['rythmLily'] = Dlugosci_wartosciRytm(nutyLib['dlugosci']) # szesnatska, 0.25

    nutyLib['generate_opis'],nutyLib['generate'] = midiRythm_toLily(nutyLib['nameLily'],nutyLib['rythmLily'],audio_file)


    # ------------------------------------------------- DO WYNIKÓW -----------------------------------------------------

    print('\nSkopiuj i generuj', nutyLib['generate_opis'])

    return nutyLib


# seg_librosa()
