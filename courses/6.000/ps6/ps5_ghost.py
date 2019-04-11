# Problem Set 5: Ghost
# Name: 
# Collaborators: 
# Time: 
#

import random

# -----------------------------------
# Helper code
# (you don't need to understand this helper code)
import string

WORDLIST_FILENAME = "words.txt"

def load_words():
    """
    Returns a list of valid words. Words are strings of lowercase letters.
    
    Depending on the size of the word list, this function may
    take a while to finish.
    """
    print("Loading word list from file...")
    # inFile: file
    inFile = open(WORDLIST_FILENAME, 'r')
    # wordlist: list of strings
    wordlist = []
    for line in inFile:
        wordlist.append(line.strip().lower())
    print("  ", len(wordlist), "words loaded.")
    return wordlist

def get_frequency_dict(sequence):
    """
    Returns a dictionary where the keys are elements of the sequence
    and the values are integer counts, for the number of times that
    an element is repeated in the sequence.

    sequence: string or list
    return: dictionary
    """
    # freqs: dictionary (element_type -> int)
    freq = {}
    for x in sequence:
        freq[x] = freq.get(x,0) + 1
    return freq


# (end of helper code)
# -----------------------------------

# Actually load the dictionary of words and point to it with 
# the wordlist variable so that it can be accessed from anywhere
# in the program.
wordlist = load_words()

# TO DO: your code begins here!
def is_valid_start_of_word(start,word_list):
    valid = False
    for word in word_list:
        if len(word) < len(start):
            continue
        if start == word[0:len(start)]:
            valid = True
            break
    return valid

def play_ghost(word_list):
    current_player = 1
    fragment = ''
    print('Welcome to Ghost')
    while (1):
        print("Player {}'s turn.".format(current_player))
        print("Current word fragment: ",fragment)
        char = input("Player {} says letter: ".format(current_player))
        char = char.upper()
        if len(char) == 1:
            if char in string.ascii_letters:
                fragment += char
                if len(fragment) <= 3:
                    if current_player == 1:
                        current_player = 2
                    else:
                        current_player = 1
                    continue
                else:
                    print(fragment.upper())
                    if is_valid_start_of_word(fragment.lower(),word_list):
                        if fragment.lower() in word_list:
                            print("Player {} wins the game!!".format(current_player))
                            break
                    else:
                        print("Player {} looses for wrong word: ".format(current_player),fragment)
                        break
            else:
                print("Player {} looses for wrong input: ".format(current_player),char)
                break
        else:
            print("Player {} looses for wrong input: ".format(current_player),char)
            break

if __name__ == '__main__':
    play_ghost(wordlist)
