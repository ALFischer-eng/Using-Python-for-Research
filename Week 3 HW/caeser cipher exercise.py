# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 11:43:10 2021

@author: Ali

Exercise 1
A cipher is a secret code for a language. In this case study, we will explore a cipher that is reported by contemporary Greek historians to have been used by Julius Caesar to send secret messages to generals during times of war.

The Caesar cipher shifts each letter of a message to another letter in the alphabet located a fixed distance from the original letter. If our encryption key were 1, we would shift h to the next letter i, i to the next letter j, and so on. If we reach the end of the alphabet, which for us is the space character, we simply loop back to a. To decode the message, we make a similar shift, except we move the same number of steps backwards in the alphabet.

Over the next five exercises, we will create our own Caesar cipher, as well as a message decoder for this cipher. In this exercise, we will define the alphabet used in the cipher.
"""

import string
alphabet = " " + string.ascii_lowercase
positions = {}
for i in alphabet:
    positions[i] = alphabet.index(i)
    
message = "hi my name is caesar"


    
encoded_message = ""

for char in message:
    pos = positions[char]
    encoded_message += alphabet[(pos+1)%27]
    


def encoding(message,key):
    encoded_message = ""
    for char in message:
        pos = positions[char]
        encoded_message += alphabet[(pos+key)%27]
    return encoded_message

print(encoding(message,3))


'''decoding the message can be used with -1*key'''

encoded_message = encoding(message,3)
print(encoding(encoded_message,-3))