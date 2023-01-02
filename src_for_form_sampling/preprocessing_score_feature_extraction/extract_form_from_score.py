# Juoyterbookで使うときは
# %load_ext music21.ipython21
# というLineが必要。
# https://web.mit.edu/music21/doc/moduleReference/moduleIpython21.html
import music21 as m21
import data_scalers_and_graph_generators as dsagg
import numpy as np


# -------------------------------------------------------------------------------------
# Functions----------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# see the pitch contents
def see_some_pitch_components_of_a_score_for_my_practice(s):
    print(type(s))
    print(s.flat.notes[0])
    print(s.flat.notes[0].quarterLength)
    print(s.flat.notes[1])
    print(s.flat.notes[1].quarterLength)
    # print (dir(s.flat.notes[0]))
    print(s.flat.notes[0].pitches)


# the number of notes in each chord (or note) (very rough implmentation)
# (just to get some random graph for my algorithmic experiment)
# I am not considering the differenth of note length of each note
# quarternotesのmelodyにたいして、eighth notesの accompanymentがついているなら、
# 裏拍で起こっているnoteの数は１つではなく2つと数えるべき。Music21はこれをきちっと数えている。
# 　つまり、今自分でやらないといけないのは、quarterlengthによって、listを調整すること。
# definitely need revision!!!!!!!! May 23, 2022
def get_number_of_notes_per_chord_ndarray(s):
    number_of_notes_per_chord_list = []
    for a_note_or_chord in s.chordify().recurse().getElementsByClass('Chord'):
        the_number_of_notes = len(a_note_or_chord.pitches)
        number_of_notes_per_chord_list.append(the_number_of_notes)

    number_of_notes_per_chord_ndarray = np.array(number_of_notes_per_chord_list)
    return number_of_notes_per_chord_ndarray


# I need to consider quarter note lenght of each note later!!!!!!!!!!!!!!!!!!!!!!
# interval vector in each chord
# I add 1 to each element of each interval vector
# (こうすることで本来はゼロも含むinterval vectorを数字で表せる。)
# どのようなintevral vectorをグラフの上の方におき、どのようなinterval vectorを下に置くのか、考えないとだめ
# interval vectorの数字の上下に音楽的な意味はないから。(ただ6進法の数字としてみて、10進法に直したら、上下に意味があるかも！！
# 近い、数字ほど、pitch setを共有しているという意味で。)
def get_interval_vector_list_from_score(s):
    interval_vector_list = []
    for a_note_or_chord in s.chordify().recurse().getElementsByClass('Chord'):
        the_interval_vector = a_note_or_chord.intervalVector
        the_interval_vector_with_1_added_to_each_element = [x + 1 for x in the_interval_vector]

        # Converting integer list to string list
        s = [str(i) for i in the_interval_vector_with_1_added_to_each_element]
        # Join list items using join()
        res = int("".join(s))
        interval_vector_list.append(res)

    interval_vector_ndarray = np.array(interval_vector_list)
    return interval_vector_ndarray


# list of the highest note of each chord
# (just to get some random graph for my algorithmic experiment)
# I am not considering the differenth of note length of each note
# 各noteのquarter lengthを考えてあげないといけない。
# 　この実装はすべてのNoteの長さが同じものとして扱っている。
def get_highest_note_list_from_score(s):
    list_of_highest_note_per_chord = []
    for a_note_or_chord in s.chordify().recurse().getElementsByClass('Chord'):
        the_pitches = [p.midi for p in a_note_or_chord.pitches]
        the_highes_pitch = max(the_pitches)
        list_of_highest_note_per_chord.append(the_highes_pitch)

    ndarray_of_highest_note_per_chord = np.array(list_of_highest_note_per_chord)
    return ndarray_of_highest_note_per_chord


def get_lowest_note_list_from_score(s):
    list_of_lowest_note_per_chord = []
    for a_note_or_chord in s.chordify().recurse().getElementsByClass('Chord'):
        the_pitches = [p.midi for p in a_note_or_chord.pitches]
        the_highes_pitch = min(the_pitches)
        list_of_lowest_note_per_chord.append(the_highes_pitch)

    ndarray_of_lowest_note_per_chord = np.array(list_of_lowest_note_per_chord)
    return ndarray_of_lowest_note_per_chord


# -------------------------------------------------------------------------------------
# Driver Codes-------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# parse musicxml file
s = m21.converter.parse('zz-scoreFiles/Chopin_Sonata_No._2_4th_Movement.mxl')

number_of_notes_per_chord_ndarray = get_number_of_notes_per_chord_ndarray(s)
# scaled_number_of_notes_per_chord_ndarray = dsagg.scale_a_ndarray(number_of_notes_per_chord_ndarray, 0, 127)
dsagg.plot_a_line_graph_from_a_vector_ndarray(number_of_notes_per_chord_ndarray, 'number_of_notes_per_chord')
np.savetxt('numberOfNotesPerChord.txt', number_of_notes_per_chord_ndarray, fmt='%1.9f')

interval_vector_ndarray_with_one_added = get_interval_vector_list_from_score(s)
scaled_interval_vector_ndarray_with_one_added = dsagg.scale_a_ndarray(interval_vector_ndarray_with_one_added, 0, 127)
dsagg.plot_a_line_graph_from_a_vector_ndarray(scaled_interval_vector_ndarray_with_one_added,
                                              'interval vectors | 1 is added to each number in each vector for numerical representation')
np.savetxt('somethingInteresting.txt', scaled_interval_vector_ndarray_with_one_added, fmt='%1.9f')

ndarray_of_highest_note_per_chord = get_highest_note_list_from_score(s)
dsagg.plot_a_line_graph_from_a_vector_ndarray(ndarray_of_highest_note_per_chord, 'highest notes')
np.savetxt('highestNotesOfEachChord.txt', ndarray_of_highest_note_per_chord, fmt='%1.9f')

ndarray_of_lowest_note_per_chord = get_lowest_note_list_from_score(s)
dsagg.plot_a_line_graph_from_a_vector_ndarray(ndarray_of_lowest_note_per_chord, 'lowest notes')
np.savetxt('lowesttNotesOfEachChord.txt', ndarray_of_lowest_note_per_chord, fmt='%1.9f')

# -------------------------------------------------------------------------------------
# toImplement--------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
'''
Refer to
https://web.mit.edu/music21/doc/moduleReference/moduleChord.html

#plot two graphs (highest and lowest notes, for example) to a single graph.

music21.chord.normalOrder
music21.chord.orderedPitchClasses
music21.chord.primeForm
music21.chord.duration
music21.chord.notes




'''
